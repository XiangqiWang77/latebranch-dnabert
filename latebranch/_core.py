"""Checkpoint-compatible DNABERT operations extracted from the study implementation.

Only model construction, shared encoder execution, context and readout operations
are retained; filesystem, scheduler and historical analysis code are excluded.
"""
from typing import Dict, Mapping, Optional, Sequence
import numpy as np
import torch
from torch import nn
N_AUTOSOMES = 22

def patch_dnabert_eager(backbone: nn.Module) -> int:
    count = 0
    for module in backbone.modules():
        if hasattr(module, "p_dropout"):
            module.p_dropout = 1e-12
            count += 1
    return count


def configure_backbone(backbone: nn.Module, freeze: bool, train_last_n: int = 0) -> int:
    """Configure full, partial, or frozen DNABERT adaptation.

    ``train_last_n > 0`` freezes embeddings and lower encoder blocks and only
    adapts the requested top blocks. A value of zero with freeze=False adapts
    the complete backbone.
    """

    if freeze and train_last_n:
        raise ValueError("--freeze-backbone and --train-backbone-layers are mutually exclusive")
    patch_dnabert_eager(backbone)
    for parameter in backbone.parameters():
        parameter.requires_grad_(not freeze and train_last_n == 0)
    if not freeze and train_last_n > 0:
        layers = getattr(getattr(backbone, "encoder", None), "layer", None)
        if layers is None:
            raise RuntimeError("Could not locate DNABERT encoder.layer for partial fine-tuning.")
        if train_last_n > len(layers):
            raise ValueError(f"Requested {train_last_n} layers, but DNABERT has {len(layers)}")
        for layer in layers[-train_last_n:]:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
    if (
        not freeze
        and train_last_n == 0
        and bool(getattr(backbone, "supports_gradient_checkpointing", False))
    ):
        backbone.gradient_checkpointing_enable()
    backbone._intervention_train_last_n = int(train_last_n)  # type: ignore[attr-defined]
    return sum(p.numel() for p in backbone.parameters() if p.requires_grad)


class BiologicalContextEncoder(nn.Module):
    """Interaction-aware assay/tissue/chromosome context for FiLM.

    Each field is embedded separately. Pairwise products make the three-way
    biological state more expressive than a plain concatenation while keeping
    donor identity and absolute genomic coordinate completely out of the model.
    """

    def __init__(
        self,
        n_tissues: int,
        n_assays: int,
        hidden: int,
        width: int = 64,
        fields: Sequence[str] = ("assay", "tissue", "chromosome"),
        use_pairwise_interactions: bool = True,
    ):
        super().__init__()
        allowed = {"assay", "tissue", "chromosome"}
        unknown = set(fields) - allowed
        if unknown or not fields:
            raise ValueError(
                f"Context fields must be a nonempty subset of {sorted(allowed)}; "
                f"got {list(fields)}"
            )
        self.fields = tuple(name for name in ("assay", "tissue", "chromosome") if name in fields)
        self.use_pairwise_interactions = bool(use_pairwise_interactions)
        self.tissue = nn.Embedding(n_tissues, width)
        self.assay = nn.Embedding(n_assays, width)
        self.chromosome = nn.Embedding(N_AUTOSOMES, width)
        self.tissue_norm = nn.LayerNorm(width)
        self.assay_norm = nn.LayerNorm(width)
        self.chromosome_norm = nn.LayerNorm(width)
        self.fusion = nn.Sequential(
            nn.LayerNorm(width * 6),
            nn.Linear(width * 6, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        tissue = self.tissue_norm(self.tissue(batch["tissue_id"]))
        assay = self.assay_norm(self.assay(batch["assay_id"]))
        chromosome = self.chromosome_norm(
            self.chromosome(batch["chromosome_id"])
        )
        if "assay" not in self.fields:
            assay = torch.zeros_like(assay)
        if "tissue" not in self.fields:
            tissue = torch.zeros_like(tissue)
        if "chromosome" not in self.fields:
            chromosome = torch.zeros_like(chromosome)
        if self.use_pairwise_interactions:
            assay_tissue = assay * tissue
            assay_chromosome = assay * chromosome
            tissue_chromosome = tissue * chromosome
        else:
            assay_tissue = torch.zeros_like(assay)
            assay_chromosome = torch.zeros_like(assay)
            tissue_chromosome = torch.zeros_like(assay)
        return self.fusion(
            torch.cat(
                [
                    assay,
                    tissue,
                    chromosome,
                    assay_tissue,
                    assay_chromosome,
                    tissue_chromosome,
                ],
                dim=-1,
            )
        )


class AlignedFourBasePotentialHead(nn.Module):
    """One shared prediction head evaluated on aligned A/C/G/T branches."""

    def __init__(
        self,
        hidden: int,
        heads: int = 8,
    ):
        super().__init__()
        self.local_attention = nn.MultiheadAttention(
            hidden, heads, dropout=0.1, batch_first=True
        )
        self.shared_scalar = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        candidate_states: torch.Tensor,
        attention_mask: torch.Tensor,
        marker_position: torch.Tensor,
    ) -> torch.Tensor:
        observations, candidates, width, hidden = candidate_states.shape
        flat = candidate_states.reshape(observations * candidates, width, hidden)
        flat_mask = attention_mask[:, None, :].expand(-1, candidates, -1).reshape(
            observations * candidates, width
        )
        flat_marker = marker_position[:, None].expand(-1, candidates).reshape(-1)
        row = torch.arange(len(flat), device=flat.device)
        center = flat[row, flat_marker]
        local, _ = self.local_attention(
            center[:, None, :],
            flat,
            flat,
            key_padding_mask=flat_mask.eq(0),
            need_weights=False,
        )
        scores = self.shared_scalar(torch.cat([center, local[:, 0]], dim=-1)).reshape(
            observations, candidates
        )
        return scores - scores.mean(dim=-1, keepdim=True)


def effects_from_potentials(scores: torch.Tensor, ref_id: torch.Tensor, alt_id: torch.Tensor) -> torch.Tensor:
    row = torch.arange(len(scores), device=scores.device)
    return scores[row, alt_id] - scores[row, ref_id]


class InterventionDNABERT(nn.Module):

    def __init__(self, backbone: nn.Module, n_tissues: int, n_assays: int, rho: np.ndarray, base_initialization: Optional[torch.Tensor]=None, use_film: bool=True, intervention_variant: str='full', branch_layer: int=8, injection_mode: str='center_delta', context_fields: Sequence[str]=('assay', 'tissue', 'chromosome'), context_pairwise: bool=True, film_initialization: str='zero', film_placement: str='each_upper'):
        super().__init__()
        self.intervention_variant = intervention_variant
        self.use_film = use_film
        self.backbone = backbone
        hidden = int(backbone.config.hidden_size)
        n_layers = len(backbone.encoder.layer)
        if not 0 <= branch_layer <= n_layers:
            raise ValueError(f'branch_layer must be in [0,{n_layers}]')
        valid_injection_modes = {'center_delta', 'center_delta_span', 'center_delta_fixed', 'center_absolute', 'center_overwrite', 'cls_delta', 'local_delta'}
        if injection_mode not in valid_injection_modes:
            raise ValueError(f'Unknown injection mode {injection_mode!r}')
        if film_initialization not in {'zero', 'random'}:
            raise ValueError(f'Unknown FiLM initialization {film_initialization!r}')
        if film_placement not in {'each_upper', 'last_upper', 'terminal'}:
            raise ValueError(f'Unknown FiLM placement {film_placement!r}')
        self.film_placement = film_placement
        if base_initialization is None or tuple(base_initialization.shape) != (4, hidden):
            raise ValueError('REF-based late branching requires four pretrained base embeddings')
        self.branch_layer = int(branch_layer)
        self.injection_mode = injection_mode
        self.base = nn.Embedding(4, hidden)
        with torch.no_grad():
            self.base.weight.copy_(base_initialization.to(self.base.weight))
        self.candidate_projection = nn.Linear(hidden, hidden, bias=False)
        nn.init.eye_(self.candidate_projection.weight)
        if injection_mode == 'center_delta_fixed':
            self.base.weight.requires_grad_(False)
            self.candidate_projection.weight.requires_grad_(False)
        self.injection_gate = nn.Parameter(torch.tensor(1.0))
        self.context_encoder = BiologicalContextEncoder(n_tissues=n_tissues, n_assays=n_assays, hidden=hidden, fields=context_fields, use_pairwise_interactions=context_pairwise)
        self.upper_film = nn.ModuleList([nn.Linear(hidden, hidden * 2) for _ in range(n_layers - branch_layer)])
        if film_initialization == 'zero':
            for film in self.upper_film:
                nn.init.zeros_(film.weight)
                nn.init.zeros_(film.bias)
        self.terminal_film = nn.Linear(hidden, hidden * 2) if branch_layer == n_layers else None
        if self.terminal_film is not None and film_initialization == 'zero':
            nn.init.zeros_(self.terminal_film.weight)
            nn.init.zeros_(self.terminal_film.bias)
        self.head = AlignedFourBasePotentialHead(hidden, heads=8)
        self.register_buffer('rho', torch.tensor(rho, dtype=torch.float32))

    def _run_encoder_span(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, start: int, end: int, condition: Optional[torch.Tensor]=None) -> torch.Tensor:
        encoder = self.backbone.encoder
        (batch, seqlen, hidden) = hidden_states.shape
        mask = attention_mask.bool()
        indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
        lengths = mask.sum(dim=-1, dtype=torch.int32)
        cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32, device=lengths.device), lengths.cumsum(0)])
        unpadded = hidden_states.reshape(batch * seqlen, hidden).index_select(0, indices)
        if encoder._current_alibi_size < seqlen:
            encoder.rebuild_alibi_tensor(size=seqlen, device=hidden_states.device)
        elif encoder.alibi.device != hidden_states.device:
            encoder.alibi = encoder.alibi.to(hidden_states.device)
        padding_bias = (1.0 - attention_mask.to(torch.float32))[:, None, None, :] * -10000.0
        attention_bias = padding_bias + encoder.alibi[:, :, :seqlen, :seqlen]
        token_rows = indices.div(seqlen, rounding_mode='floor')
        for (layer_index, layer) in enumerate(encoder.layer[start:end]):
            unpadded = layer(unpadded, cu_seqlens, seqlen, None, indices, attn_mask=attention_mask, bias=attention_bias)
            if condition is not None and self.use_film:
                apply_here = self.film_placement == 'each_upper' or (self.film_placement == 'last_upper' and layer_index == len(self.upper_film) - 1)
                if apply_here:
                    (gamma, beta) = self.upper_film[layer_index](condition).chunk(2, dim=-1)
                    unpadded = unpadded * (1.0 + 0.1 * torch.tanh(gamma[token_rows])) + beta[token_rows]
        padded = hidden_states.new_zeros(batch * seqlen, hidden).index_copy(0, indices, unpadded)
        return padded.reshape(batch, seqlen, hidden)

    def lower_reference_states(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Run natural REF through the shared lower DNABERT trunk exactly once."""
        embedded = self.backbone.embeddings(input_ids=batch['input_ids'], token_type_ids=torch.zeros_like(batch['input_ids']))
        return self._run_encoder_span(embedded, batch['attention_mask'], 0, self.branch_layer)

    def condition_state(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.context_encoder(batch)


class SiameseDNABERT(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        n_tissues: int,
        n_assays: int,
        rho: np.ndarray,
        context_fields: Sequence[str] = ("assay", "tissue", "chromosome"),
        context_pairwise: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        hidden = int(backbone.config.hidden_size)
        self.context_encoder = BiologicalContextEncoder(
            n_tissues=n_tissues,
            n_assays=n_assays,
            hidden=hidden,
            fields=context_fields,
            use_pairwise_interactions=context_pairwise,
        )
        # Match the Intervention head's capacity (~4M parameters) so the
        # comparison isolates late branching and upper-layer FiLM rather than
        # giving either model additional biological covariates.
        self.potential = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, 2304),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2304, 1152),
            nn.GELU(),
            nn.Linear(1152, 1),
        )
        self.register_buffer("rho", torch.tensor(rho, dtype=torch.float32))

    @staticmethod
    def pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weight).sum(1) / weight.sum(1).clamp_min(1)

    def score(
        self,
        state: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        context = self.context_encoder(batch)
        return self.potential(torch.cat([state, context], dim=-1)).squeeze(-1)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ref_hidden = self.backbone(
            input_ids=batch["ref_input_ids"], attention_mask=batch["ref_attention_mask"]
        )[0]
        alt_hidden = self.backbone(
            input_ids=batch["alt_input_ids"], attention_mask=batch["alt_attention_mask"]
        )[0]
        inverse = batch["variant_inverse"]
        ref = self.pool(ref_hidden, batch["ref_attention_mask"])[inverse]
        alt = self.pool(alt_hidden, batch["alt_attention_mask"])[inverse]
        s_ref = self.score(ref, batch)
        s_alt = self.score(alt, batch)
        return {"tau": s_alt - s_ref, "rho": self.rho[batch["assay_id"]]}
