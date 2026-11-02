"""Tests for the ModernProst pLDDT head (pholdlib.modernprost.plddt).

A tiny stand-in backbone (``model.encoder.layers`` + ``model.encoder(ids, mask)``,
the only parts the head touches) and a randomly initialised head saved in the
Hugging Face repo layout (``<dir>/plddt_head/head.safetensors`` +
``head_config.json``) keep these tests offline and fast. The real heads are
checked against the training script in the release notes, not here.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from pholdlib.modernprost.plddt import PlddtHead, load_plddt_head, predict_plddt  # noqa: E402

HIDDEN = 16
N_LAYERS = 3


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(32, HIDDEN)
        self.layers = nn.ModuleList(nn.Linear(HIDDEN, HIDDEN) for _ in range(N_LAYERS))

    def forward(self, input_ids, attention_mask):
        h = self.emb(input_ids)
        for layer in self.layers:
            h = torch.tanh(layer(h))
        return h


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=HIDDEN)
        self.encoder = _Encoder()


class _Tokenizer:
    """One token per residue, right-padded with 0 — like the ModernProst tokenizer."""

    def __call__(self, seqs, add_special_tokens=False, padding=True, return_tensors="pt"):
        L = max(len(s) for s in seqs)
        ids = torch.zeros(len(seqs), L, dtype=torch.long)
        mask = torch.zeros(len(seqs), L, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor([ord(c) % 31 + 1 for c in s])
            mask[i, : len(s)] = 1
        return {"input_ids": ids, "attention_mask": mask}


@pytest.fixture()
def repo(tmp_path):
    torch.manual_seed(0)
    head = PlddtHead(HIDDEN, N_LAYERS, d=8, kernel=3, n_blocks=1, n_bins=10)
    d = tmp_path / "modernprost-tiny" / "plddt_head"
    d.mkdir(parents=True)
    save_file({k: v.contiguous() for k, v in head.state_dict().items()}, str(d / "head.safetensors"))
    (d / "head_config.json").write_text(json.dumps({
        "model": "modernprost-tiny", "layers": list(range(1, N_LAYERS + 1)), "hidden_size": HIDDEN,
        "head_dim": 8, "kernel": 3, "n_blocks": 1, "n_bins": 10, "n_attn_layers": 0, "attn_heads": 4,
    }))
    return tmp_path / "modernprost-tiny"


SEQS = {"a": "MKTAYIAKQR", "b": "MSTNPKPQRKTKRNTNRRPQDVKFPGG", "c": "M"}


def test_load_from_local_repo_dir(repo):
    head = load_plddt_head(str(repo), model_dir=repo, device=torch.device("cpu"))
    assert head.layers == [1, 2, 3]
    assert not head.training


def test_predictions_cover_every_residue_in_range(repo):
    dev = torch.device("cpu")
    head = load_plddt_head(str(repo), model_dir=repo, device=dev)
    out = predict_plddt(_Model().eval(), _Tokenizer(), head, SEQS, dev)
    assert set(out) == set(SEQS)
    for k, s in SEQS.items():
        assert out[k].shape == (len(s),)
        assert ((out[k] >= 0) & (out[k] <= 100)).all()


def test_batching_and_padding_do_not_change_predictions(repo):
    dev = torch.device("cpu")
    head = load_plddt_head(str(repo), model_dir=repo, device=dev)
    model = _Model().eval()
    together = predict_plddt(model, _Tokenizer(), head, SEQS, dev)
    for k, s in SEQS.items():
        alone = predict_plddt(model, _Tokenizer(), head, {k: s}, dev)[k]
        np.testing.assert_allclose(together[k], alone, atol=1e-4)


def test_windows_are_stitched_in_order(repo):
    dev = torch.device("cpu")
    head = load_plddt_head(str(repo), model_dir=repo, device=dev)
    model = _Model().eval()
    seq = SEQS["b"]  # 27 residues -> windows of 10, 10, 7
    windowed = predict_plddt(model, _Tokenizer(), head, {"b": seq}, dev, chunk_len=10)["b"]
    slices = {str(i): seq[i : i + 10] for i in range(0, len(seq), 10)}
    direct = predict_plddt(model, _Tokenizer(), head, slices, dev)
    np.testing.assert_allclose(windowed, np.concatenate([direct[str(i)] for i in range(0, len(seq), 10)]),
                               atol=1e-5)


def test_hooks_are_removed(repo):
    dev = torch.device("cpu")
    head = load_plddt_head(str(repo), model_dir=repo, device=dev)
    model = _Model().eval()
    predict_plddt(model, _Tokenizer(), head, SEQS, dev)
    assert all(not layer._forward_hooks for layer in model.encoder.layers)


def test_head_must_match_backbone_width(repo):
    dev = torch.device("cpu")
    head = load_plddt_head(str(repo), model_dir=repo, device=dev)
    model = _Model().eval()
    model.config.hidden_size = HIDDEN * 2
    with pytest.raises(ValueError, match="hidden size"):
        predict_plddt(model, _Tokenizer(), head, SEQS, dev)
