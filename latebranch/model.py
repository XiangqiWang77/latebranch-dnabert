"""Late branching with localized allele queries and matched terminal controls."""
import torch
from torch import nn
from ._core import InterventionDNABERT, effects_from_potentials


class LateBranch(InterventionDNABERT):
    def __init__(self, *args, neutral_anchor=False, position_aware=False, **kwargs):
        kwargs.setdefault("context_fields", ("assay", "tissue"))
        super().__init__(*args, film_placement="terminal", **kwargs)
        self.neutral_anchor = neutral_anchor
        self.position_aware = position_aware
        hidden = self.base.embedding_dim
        # One identical FiLM at the readout for both branch=8 and branch=12.
        # Remove unused FiLM parameters rather than counting dead capacity.
        self.upper_film = nn.ModuleList()
        self.terminal_film = nn.Linear(hidden, 2 * hidden)
        nn.init.zeros_(self.terminal_film.weight)
        nn.init.zeros_(self.terminal_film.bias)
        # No independent candidate dropout: it can overwhelm small score differences.
        self.head.local_attention.dropout = 0.
        self.position_gate = nn.Linear(4, hidden, bias=False) if position_aware else None
        if self.position_gate is not None:
            nn.init.zeros_(self.position_gate.weight)

    def candidate_update(self, ref_id, span, offset):
        if self.neutral_anchor:
            # The input anchor and all four potentials are independent of allele roles.
            source = self.base.weight - self.base.weight.mean(dim=0, keepdim=True)
            source = source[None].expand(len(ref_id), -1, -1)
        else:
            source = self.base.weight[None] - self.base(ref_id)[:, None]
        update = self.candidate_projection(source)
        if self.position_gate is not None:
            length = span.float().clamp_min(1)
            position = offset.float()
            features = torch.stack([position / length, (length - 1 - position) / length,
                                    1 / length, length.log() / 5], dim=-1)
            update = update * (1 + .5 * torch.tanh(self.position_gate(features))[:, None])
        return self.injection_gate * update

    def potential_components(self, batch, condition_metadata=True, block_nonfocal=False):
        inverse = batch["variant_inverse"]
        lower = self.lower_reference_states(batch)[inverse]
        mask = batch["attention_mask"][inverse]
        marker = batch["marker_position"][inverse]
        span = batch["marker_span_length"][inverse]
        offset = batch["marker_offset"][inverse]
        observations, width, hidden = lower.shape
        branches = lower[:, None].expand(-1, 4, -1, -1).clone()
        row = torch.arange(observations, device=lower.device)[:, None]
        candidate = torch.arange(4, device=lower.device)[None]
        branches[row, candidate, marker[:, None]] += self.candidate_update(batch["ref_id"], span, offset)
        branch_mask = mask[:, None].expand(-1, 4, -1).reshape(-1, width)
        upper = branches.reshape(-1, width, hidden)
        final_layer = len(self.backbone.encoder.layer)
        if block_nonfocal:
            # Causal *model* intervention: after every upper layer, replace each
            # nonfocal candidate state by the candidate mean. The focal path remains.
            # This is not a biological intervention or a retrained ablation.
            keep = torch.nn.functional.one_hot(marker, width).bool()[:, None, :, None]
            for layer in range(self.branch_layer, final_layer):
                upper = self._run_encoder_span(upper, branch_mask, layer, layer + 1)
                shaped = upper.reshape(observations, 4, width, hidden)
                upper = torch.where(keep, shaped, shaped.mean(dim=1, keepdim=True)).reshape(-1, width, hidden)
        elif self.branch_layer < final_layer:
            upper = self._run_encoder_span(upper, branch_mask, self.branch_layer, final_layer)
        if condition_metadata and self.use_film:
            condition = self.condition_state(batch).repeat_interleave(4, dim=0)
            gamma, beta = self.terminal_film(condition).chunk(2, dim=-1)
            upper = upper * (1 + .1 * torch.tanh(gamma[:, None])) + beta[:, None]
        upper = upper.reshape(observations, 4, width, hidden)
        return {"total": self.head(upper, mask, marker), "shared_lower": lower,
                "candidate_upper": upper, "observation_marker": marker}

    def forward(self, batch, block_nonfocal=False):
        scores = self.potential_components(batch, block_nonfocal=block_nonfocal)["total"]
        return {"tau": effects_from_potentials(scores, batch["ref_id"], batch["alt_id"]),
                "scores": scores, "rho": self.rho[batch["assay_id"]]}
