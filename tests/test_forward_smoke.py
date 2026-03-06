"""Forward smoke tests for one-stage and two-stage patcher pipelines."""

from copy import deepcopy

import pytest

from llm_lab.config.defaults import DEFAULT_CONFIG
from llm_lab.config.schema import ComponentCfg
from llm_lab.registry import register_mixer, register_patcher




@register_mixer("identity_probe")
class IdentityProbeMixer(__import__("torch").nn.Module):
    """Mixer that records its latest input sequence length."""

    last_input_len = -1

    def __init__(self, d_model: int = 16) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, h):
        type(self).last_input_len = int(h.shape[1])
        return h

@register_patcher("hidden_pool")
class HiddenPoolPatcher:
    """Simple hidden->hidden patcher for pipeline smoke testing."""

    def __init__(self, pool: int = 2) -> None:
        self.pool = pool

    def forward(self, h):
        torch = __import__("torch")
        if h.ndim != 3:
            raise ValueError("hidden_pool expects [B, T, D]")
        bsz, t, d = h.shape
        pad = (self.pool - (t % self.pool)) % self.pool
        if pad:
            h = torch.nn.functional.pad(h, (0, 0, 0, pad), value=0.0)
        tp = h.shape[1] // self.pool
        return h.view(bsz, tp, self.pool, d).mean(dim=2)



@register_patcher("hidden_linear")
class HiddenLinearPatcher:
    """Hidden->hidden patcher with parameters for freeze tests."""

    def __init__(self, d_model: int = 16) -> None:
        torch = __import__("torch")
        self.proj = torch.nn.Linear(d_model, d_model)

    def parameters(self):
        return self.proj.parameters()

    def forward(self, h):
        return self.proj(h)


@register_patcher("raw_only")
class RawOnlyPatcher:
    """Raw-only patcher to verify patcher2 hidden-input contract is enforced."""

    def forward(self, x_u8):
        torch = __import__("torch")
        if x_u8.dtype != torch.uint8 or x_u8.ndim != 2:
            raise ValueError("raw_only expects uint8 [B, T]")
        return x_u8.to(torch.float32).unsqueeze(-1)



def _base_cfg():
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg.model.codec.name = "identity"
    cfg.model.codec.kwargs = {}
    cfg.model.patcher1.name = "chunk"
    cfg.model.patcher1.kwargs = {"patch_size": 8, "d_model": 16}
    cfg.model.head.name = "byte"
    cfg.model.head.kwargs = {"d_model": 16}
    cfg.model.mixers = [
        ComponentCfg(name="transformer", kwargs={"d_model": 16, "n_heads": 4, "n_layers": 1})
    ]
    cfg.model.mixer = cfg.model.mixers[0]
    return cfg


def test_forward_smoke_one_stage_patcher() -> None:
    torch = pytest.importorskip("torch")
    from llm_lab.models.assemble import assemble_model

    cfg = _base_cfg()
    cfg.model.patcher2 = None

    model = assemble_model(cfg)
    x = torch.randint(0, 256, (2, 17), dtype=torch.uint8)
    logits = model(x)

    # patcher1 with patch_size=8 gives T1=3
    assert logits.shape == (2, 3, 256)


def test_forward_smoke_two_stage_patchers() -> None:
    torch = pytest.importorskip("torch")
    from llm_lab.models.assemble import assemble_model

    cfg = _base_cfg()
    cfg.model.patcher2 = ComponentCfg(
        name="hidden_chunk", kwargs={"patch_size": 2, "d_model": 16, "use_mlp": False}
    )

    model = assemble_model(cfg)
    x = torch.randint(0, 256, (2, 17), dtype=torch.uint8)
    logits = model(x)

    # T1=3 from chunk patcher, then hidden_chunk(patch_size=2, drop remainder) -> T2=1
    assert logits.shape == (2, 1, 256)
    assert logits.shape[1] < 3


def test_forward_fails_if_patcher2_expects_raw_bytes() -> None:
    torch = pytest.importorskip("torch")
    from llm_lab.models.assemble import assemble_model

    cfg = _base_cfg()
    cfg.model.patcher2 = ComponentCfg(name="raw_only", kwargs={})

    model = assemble_model(cfg)
    x = torch.randint(0, 256, (2, 17), dtype=torch.uint8)

    with pytest.raises(ValueError, match="raw_only expects uint8"):
        model(x)


def test_freeze_patcher1_reduces_trainable_params() -> None:
    pytest.importorskip("torch")
    from llm_lab.models.assemble import assemble_model

    cfg_a = _base_cfg()
    cfg_b = _base_cfg()
    cfg_b.model.patcher1.freeze = True

    model_a = assemble_model(cfg_a)
    model_b = assemble_model(cfg_b)

    c_a = model_a.component_param_counts()
    c_b = model_b.component_param_counts()

    assert c_b["patcher1"]["trainable"] < c_a["patcher1"]["trainable"]
    assert c_b["patcher1"]["trainable"] == 0


def test_freeze_patcher2_reduces_trainable_params() -> None:
    pytest.importorskip("torch")
    from llm_lab.models.assemble import assemble_model

    cfg_a = _base_cfg()
    cfg_a.model.patcher2 = ComponentCfg(name="hidden_linear", kwargs={"d_model": 16})

    cfg_b = _base_cfg()
    cfg_b.model.patcher2 = ComponentCfg(name="hidden_linear", kwargs={"d_model": 16}, freeze=True)

    model_a = assemble_model(cfg_a)
    model_b = assemble_model(cfg_b)

    c_a = model_a.component_param_counts()
    c_b = model_b.component_param_counts()

    assert c_b["patcher2"]["trainable"] <= c_a["patcher2"]["trainable"]
    assert c_b["patcher2"]["trainable"] == 0


def test_freeze_both_patchers_independent() -> None:
    pytest.importorskip("torch")
    from llm_lab.models.assemble import assemble_model

    cfg = _base_cfg()
    cfg.model.patcher1.freeze = True
    cfg.model.patcher2 = ComponentCfg(name="hidden_linear", kwargs={"d_model": 16}, freeze=True)

    model = assemble_model(cfg)
    counts = model.component_param_counts()

    assert counts["patcher1"]["trainable"] == 0
    assert counts["patcher2"]["trainable"] == 0


def test_forward_smoke_with_aux_reconstruction_logits() -> None:
    torch = pytest.importorskip("torch")
    from llm_lab.models.assemble import assemble_model

    cfg = _base_cfg()
    cfg.train.enable_aux_reconstruction = True
    cfg.train.aux_reconstruction_weight = 0.1

    model = assemble_model(cfg)
    x = torch.randint(0, 256, (2, 17), dtype=torch.uint8)

    logits, recon_logits = model.forward_with_aux(x)

    assert logits.shape == (2, 3, 256)
    assert recon_logits is not None
    assert recon_logits.shape == (2, 3, 8, 256)


def test_train_loop_aux_reconstruction_loss_finite() -> None:
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    from llm_lab.models.assemble import assemble_model
    from llm_lab.train.loop import train_loop
    from llm_lab.train.optim import build_optimizer

    cfg = _base_cfg()
    cfg.train.enable_aux_reconstruction = True
    cfg.train.aux_reconstruction_weight = 0.1

    model = assemble_model(cfg)
    optimizer = build_optimizer(model, lr=1e-3)

    samples = [torch.randint(0, 256, (17,), dtype=torch.uint8) for _ in range(4)]
    dataloader = DataLoader(samples, batch_size=2, shuffle=False)

    metrics = train_loop(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        steps=1,
        enable_aux_reconstruction=True,
        aux_reconstruction_weight=cfg.train.aux_reconstruction_weight,
    )

    assert metrics["step"] == 1
    assert float(metrics["loss"]) == pytest.approx(float(metrics["loss"]))



def test_context_pyramid_smoke_patcher1_only() -> None:
    torch = pytest.importorskip("torch")
    from llm_lab.models.assemble import assemble_model

    cfg = _base_cfg()
    cfg.model.patcher2 = None
    cfg.model.mixers = [ComponentCfg(name="identity_probe", kwargs={"d_model": 16})]
    cfg.model.mixer = cfg.model.mixers[0]
    cfg.model.memory = ComponentCfg(
        name="context_pyramid",
        kwargs={
            "keep_recent_tokens": 2,
            "keep_lowres_history": 4,
            "lowres_source": "pooled_patcher1",
        },
    )

    model = assemble_model(cfg)

    x1 = torch.randint(0, 256, (2, 41), dtype=torch.uint8)
    logits1 = model(x1)
    mix_len1 = IdentityProbeMixer.last_input_len

    x2 = torch.randint(0, 256, (2, 41), dtype=torch.uint8)
    logits2 = model(x2)
    mix_len2 = IdentityProbeMixer.last_input_len

    assert logits1.shape[1] == 2
    assert logits2.shape[1] == 2
    assert mix_len1 > logits1.shape[1]
    assert mix_len2 > logits2.shape[1]


def test_context_pyramid_smoke_patcher1_and_patcher2() -> None:
    torch = pytest.importorskip("torch")
    from llm_lab.models.assemble import assemble_model

    cfg = _base_cfg()
    cfg.model.patcher2 = ComponentCfg(
        name="hidden_chunk", kwargs={"patch_size": 2, "d_model": 16, "use_mlp": False}
    )
    cfg.model.mixers = [ComponentCfg(name="identity_probe", kwargs={"d_model": 16})]
    cfg.model.mixer = cfg.model.mixers[0]
    cfg.model.memory = ComponentCfg(
        name="context_pyramid",
        kwargs={
            "keep_recent_tokens": 2,
            "keep_lowres_history": 3,
            "lowres_source": "patcher2",
        },
    )

    model = assemble_model(cfg)

    x = torch.randint(0, 256, (2, 41), dtype=torch.uint8)
    logits = model(x)
    mix_len = IdentityProbeMixer.last_input_len

    assert logits.shape[1] == 2
    assert mix_len > logits.shape[1]



def test_dynamic_blt_fixed_smoke() -> None:
    torch = pytest.importorskip("torch")
    from llm_lab.models.assemble import assemble_model

    cfg = _base_cfg()
    cfg.model.patcher1 = ComponentCfg(
        name="dynamic_blt",
        kwargs={"max_patch_size": 8, "min_patch_size": 2, "d_model": 16, "heuristic": "fixed"},
    )
    cfg.model.patcher2 = None

    model = assemble_model(cfg)
    x = torch.randint(0, 256, (2, 24), dtype=torch.uint8)
    logits = model(x)

    assert logits.shape == (2, 3, 256)


def test_dynamic_blt_whitespace_smoke_and_count_change() -> None:
    torch = pytest.importorskip("torch")
    from llm_lab.models.assemble import assemble_model

    cfg_fixed = _base_cfg()
    cfg_fixed.model.patcher1 = ComponentCfg(
        name="dynamic_blt",
        kwargs={"max_patch_size": 8, "min_patch_size": 2, "d_model": 16, "heuristic": "fixed"},
    )
    cfg_fixed.model.patcher2 = None

    cfg_ws = _base_cfg()
    cfg_ws.model.patcher1 = ComponentCfg(
        name="dynamic_blt",
        kwargs={"max_patch_size": 8, "min_patch_size": 2, "d_model": 16, "heuristic": "whitespace"},
    )
    cfg_ws.model.patcher2 = None

    model_fixed = assemble_model(cfg_fixed)
    model_ws = assemble_model(cfg_ws)

    text = b"hello world this is dynamic blt"
    x = torch.tensor([list(text)], dtype=torch.uint8)

    logits_fixed = model_fixed(x)
    logits_ws = model_ws(x)

    assert logits_ws.shape[2] == 256
    assert logits_fixed.shape[2] == 256
    assert logits_fixed.shape[1] != logits_ws.shape[1]



def test_fsq_codec_quantize_shape_smoke() -> None:
    torch = pytest.importorskip("torch")
    from llm_lab.components.codecs.fsq import FSQCodec

    codec = FSQCodec(levels_per_dim=8, d_model=16)
    h = torch.randn(2, 5, 16)
    q = codec.quantize_hidden(h)

    assert q.shape == h.shape


def test_fsq_codec_quantization_changes_values() -> None:
    torch = pytest.importorskip("torch")
    from llm_lab.components.codecs.fsq import FSQCodec

    codec = FSQCodec(levels_per_dim=4, d_model=16)
    h = torch.linspace(-0.9, 0.9, steps=2 * 5 * 16, dtype=torch.float32).view(2, 5, 16)
    q = codec.quantize_hidden(h)

    assert not torch.allclose(q, h)


def test_fsq_codec_straight_through_gradients() -> None:
    torch = pytest.importorskip("torch")
    from llm_lab.components.codecs.fsq import FSQCodec

    codec = FSQCodec(levels_per_dim=8, d_model=16)
    h = torch.randn(2, 5, 16, requires_grad=True)
    q = codec.quantize_hidden(h)
    loss = q.sum()
    loss.backward()

    assert h.grad is not None
    assert torch.isfinite(h.grad).all()


def test_forward_smoke_with_fsq_codec() -> None:
    torch = pytest.importorskip("torch")
    from llm_lab.models.assemble import assemble_model

    cfg = _base_cfg()
    cfg.model.codec = ComponentCfg(name="fsq", kwargs={"levels_per_dim": 8, "d_model": 16})
    cfg.model.patcher2 = None

    model = assemble_model(cfg)
    x = torch.randint(0, 256, (2, 17), dtype=torch.uint8)
    logits = model(x)

    assert logits.shape == (2, 3, 256)



def test_vq_lite_quantization_smoke() -> None:
    torch = pytest.importorskip("torch")
    from llm_lab.components.codecs.vq_lite import VQLiteCodec

    codec = VQLiteCodec(codebook_size=16, d_model=8)
    h = torch.randn(2, 5, 8)
    q, idx = codec.quantize_hidden_with_codes(h)

    assert q.shape == h.shape
    assert idx.shape == (2, 5)
    assert not torch.allclose(q, h)


def test_vq_lite_code_indices_range() -> None:
    torch = pytest.importorskip("torch")
    from llm_lab.components.codecs.vq_lite import VQLiteCodec

    codec = VQLiteCodec(codebook_size=7, d_model=8)
    h = torch.randn(2, 5, 8)
    _, idx = codec.quantize_hidden_with_codes(h)

    assert int(idx.min().item()) >= 0
    assert int(idx.max().item()) < 7


def test_forward_smoke_with_vq_lite_codec() -> None:
    torch = pytest.importorskip("torch")
    from llm_lab.models.assemble import assemble_model

    cfg = _base_cfg()
    cfg.model.codec = ComponentCfg(name="vq_lite", kwargs={"codebook_size": 32, "d_model": 16})
    cfg.model.patcher2 = None

    model = assemble_model(cfg)
    x = torch.randint(0, 256, (2, 17), dtype=torch.uint8)
    logits = model(x)

    assert logits.shape == (2, 3, 256)
    assert model.last_codec_codes is not None
    assert model.last_codec_codes.shape == (2, 3)



def test_vq_lite_ema_smoke_runs() -> None:
    torch = pytest.importorskip("torch")
    from llm_lab.components.codecs.vq_lite import VQLiteCodec

    codec = VQLiteCodec(codebook_size=16, d_model=8, use_ema_updates=True, ema_decay=0.9)
    codec.train()
    h = torch.randn(2, 5, 8)
    q, idx = codec.quantize_hidden_with_codes(h)

    assert q.shape == h.shape
    assert idx.shape == (2, 5)


def test_vq_lite_ema_codebook_updates_over_forwards() -> None:
    torch = pytest.importorskip("torch")
    from llm_lab.components.codecs.vq_lite import VQLiteCodec

    codec = VQLiteCodec(codebook_size=8, d_model=4, use_ema_updates=True, ema_decay=0.5)
    codec.train()

    before = codec.codebook.weight.detach().clone()
    h = torch.randn(2, 6, 4)

    for _ in range(3):
        codec.quantize_hidden_with_codes(h)

    after = codec.codebook.weight.detach().clone()
    assert not torch.allclose(before, after)



def test_patcher_stop_gradient_blocks_upstream_grads() -> None:
    torch = pytest.importorskip("torch")
    from llm_lab.models.assemble import assemble_model

    x = torch.randint(0, 256, (2, 17), dtype=torch.uint8)

    cfg_a = _base_cfg()
    cfg_a.model.patcher2 = None
    cfg_a.model.patcher1.stop_gradient = False

    cfg_b = _base_cfg()
    cfg_b.model.patcher2 = None
    cfg_b.model.patcher1.stop_gradient = True

    model_a = assemble_model(cfg_a)
    model_b = assemble_model(cfg_b)

    logits_a = model_a(x)
    loss_a = logits_a.sum()
    loss_a.backward()

    logits_b = model_b(x)
    loss_b = logits_b.sum()
    loss_b.backward()

    grad_a = model_a.patcher1.proj.weight.grad
    grad_b = model_b.patcher1.proj.weight.grad

    assert grad_a is not None
    assert torch.isfinite(grad_a).all()
    assert grad_a.abs().sum().item() > 0
    assert grad_b is None or grad_b.abs().sum().item() == 0



def test_build_optimizer_patcher_only_includes_only_patcher_params() -> None:
    pytest.importorskip("torch")
    from llm_lab.models.assemble import assemble_model
    from llm_lab.train.optim import build_optimizer

    cfg = _base_cfg()
    cfg.model.patcher2 = ComponentCfg(name="hidden_linear", kwargs={"d_model": 16})

    model = assemble_model(cfg)
    groups = model.component_param_groups()

    optimizer = build_optimizer(model, lr=1e-3, mode="patcher_only")

    opt_param_ids = {
        id(p)
        for group in optimizer.param_groups
        for p in group["params"]
    }
    patcher_ids = {id(p) for p in groups["patcher1"] + groups["patcher2"]}
    non_patcher_ids = {
        id(p)
        for name, params in groups.items()
        if name not in {"patcher1", "patcher2"}
        for p in params
    }

    assert opt_param_ids
    assert opt_param_ids.issubset(patcher_ids)
    assert opt_param_ids.isdisjoint(non_patcher_ids)
