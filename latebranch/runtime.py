"""Pinned backbone loading and portable, tensor-only study checkpoints."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ._core import SiameseDNABERT, configure_backbone
from .model import LateBranch

MANIFEST = json.loads(Path(__file__).with_name("backbone_manifest.json").read_text())


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_backbone(path):
    path = Path(path)
    for name, record in MANIFEST["files"].items():
        file = path / name
        if not file.is_file() or file_sha256(file) != record["sha256"]:
            raise ValueError(f"Missing or changed pinned DNABERT-2 asset: {file}")


def download_backbone(path):
    from huggingface_hub import hf_hub_download
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for name in [*MANIFEST["files"], "LICENSE"]:
        existing = path / name
        if name in MANIFEST["files"] and existing.exists():
            if file_sha256(existing) != MANIFEST["files"][name]["sha256"]:
                raise ValueError(f"Refusing to replace a different asset: {existing}")
            continue
        hf_hub_download(MANIFEST["repo_id"], name, revision=MANIFEST["revision"], local_dir=str(path))
    verify_backbone(path)
    return path


@dataclass
class ModelConfig:
    assays: list[str]
    tissues: list[str]
    rho: list[float]
    model_type: str = "latebranch"
    branch_layer: int = 8
    position_aware: bool = True
    train_last_n: int = 4
    max_tokens: int = 128
    seed: int = 13

    def __post_init__(self):
        if self.model_type not in {"latebranch", "vanilla"}:
            raise ValueError("model_type must be latebranch or vanilla")
        for name in ("assays", "tissues"):
            values = getattr(self, name)
            if not values or len(values) != len(set(values)):
                raise ValueError(f"Nonempty, unique {name} names are required")
        if len(self.rho) != len(self.assays) or not all(0 < x < 1 for x in self.rho):
            raise ValueError("One training-fitted dispersion in (0,1) is required per assay")
        if not 0 <= self.branch_layer <= 12 or not 0 <= self.train_last_n <= 12:
            raise ValueError("DNABERT-2 has 12 encoder layers")
        if self.max_tokens < 3:
            raise ValueError("The token budget must include CLS, the focal token and SEP")


def build_model(config, backbone_path):
    """Build the released architecture; no data archive or scheduler is consulted."""
    from transformers import AutoModel, AutoTokenizer
    if isinstance(config, dict):
        config = ModelConfig(**config)
    verify_backbone(backbone_path)
    tokenizer = AutoTokenizer.from_pretrained(str(backbone_path), trust_remote_code=True, local_files_only=True)
    backbone = AutoModel.from_pretrained(str(backbone_path), trust_remote_code=True, local_files_only=True)
    configure_backbone(backbone, freeze=False, train_last_n=config.train_last_n)
    common = dict(n_tissues=len(config.tissues), n_assays=len(config.assays),
                  rho=np.asarray(config.rho), context_fields=("assay", "tissue"))
    if config.model_type == "vanilla":
        model = SiameseDNABERT(backbone, **common)
    else:
        base_ids = [tokenizer.convert_tokens_to_ids(base) for base in "ACGT"]
        for base, index in zip("ACGT", base_ids):
            if tokenizer(base, add_special_tokens=False)["input_ids"] != [index]:
                raise ValueError(f"No single-token nucleotide grounding for {base}")
        initial = backbone.get_input_embeddings().weight[base_ids].detach().clone()
        # Common construction preserves identical initialization across depths.
        model = LateBranch(backbone, base_initialization=initial, branch_layer=8,
                           position_aware=config.position_aware, **common)
        model.branch_layer = config.branch_layer
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.
    model.rho = torch.as_tensor(config.rho, dtype=torch.float64)
    return model, tokenizer


def adapted_keys(model, config):
    layers = len(model.backbone.encoder.layer)
    prefixes = tuple(f"backbone.encoder.layer.{i}." for i in range(layers-config.train_last_n,layers))
    parameters = {name for name,_ in model.named_parameters()
                  if not name.startswith("backbone.") or config.train_last_n == 0 or name.startswith(prefixes)}
    buffers = {name for name,_ in model.named_buffers() if not name.startswith("backbone.")}
    return parameters | buffers


def save_checkpoint(path, model, config, **metadata):
    keys = adapted_keys(model,config)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if name in keys}
    torch.save({"format_version": 1, "state_kind": "adapted", "state_dict": state, "config": asdict(config),
                "backbone_revision": MANIFEST["revision"], "metadata": metadata}, path)


def load_checkpoint(path, backbone_path, device="cpu"):
    # The release contains only tensors and primitive configuration values.
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format_version") != 1 or saved["backbone_revision"] != MANIFEST["revision"]:
        raise ValueError("Unsupported checkpoint format or backbone revision")
    config = ModelConfig(**saved["config"])
    model, tokenizer = build_model(config, backbone_path)
    if saved.get("state_kind") != "adapted" or set(saved["state_dict"]) != adapted_keys(model,config):
        raise ValueError("Adapted checkpoint parameters are incomplete or unexpected")
    incompatible=model.load_state_dict(saved["state_dict"],strict=False)
    if incompatible.unexpected_keys or any(not name.startswith("backbone.") for name in incompatible.missing_keys):
        raise ValueError("Checkpoint architecture mismatch")
    if not np.array_equal(model.rho.cpu().numpy(),config.rho):
        raise ValueError("Checkpoint configuration and saved dispersion differ")
    model.to(device).eval()
    return model, tokenizer, config
