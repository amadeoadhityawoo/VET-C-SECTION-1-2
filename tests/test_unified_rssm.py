import torch

from vetc.rssm.unified_rssm import UnifiedRSSM


def test_unified_rssm_runs_with_explicit_previous_states():
    rssm = UnifiedRSSM()
    fused = torch.randn(2, 256)
    h_prev = torch.randn(2, 512)
    z_prev = torch.randn(2, 64)

    output = rssm(fused, h_prev=h_prev, z_prev=z_prev)

    assert output["h"].shape == (2, 512)
    assert output["z_prior"].shape == (2, 64)
    assert output["z_post"].shape == (2, 64)
    assert output["prior_mu"].shape == (2, 64)
    assert output["prior_logvar"].shape == (2, 64)
    assert output["post_mu"].shape == (2, 64)
    assert output["post_logvar"].shape == (2, 64)
    for tensor in output.values():
        assert torch.isfinite(tensor).all().item()


def test_unified_rssm_runs_with_none_states():
    rssm = UnifiedRSSM()
    fused = torch.randn(2, 256)

    output = rssm(fused, h_prev=None, z_prev=None)

    assert output["h"].shape == (2, 512)
    assert output["z_prior"].shape == (2, 64)
    assert output["z_post"].shape == (2, 64)
    for tensor in output.values():
        assert torch.isfinite(tensor).all().item()


def test_unified_rssm_initial_state_returns_zero_finite_tensors():
    rssm = UnifiedRSSM()

    h_prev, z_prev = rssm.initial_state(2)

    assert h_prev.shape == (2, 512)
    assert z_prev.shape == (2, 64)
    assert torch.equal(h_prev, torch.zeros(2, 512))
    assert torch.equal(z_prev, torch.zeros(2, 64))
    assert torch.isfinite(h_prev).all().item()
    assert torch.isfinite(z_prev).all().item()


def test_unified_rssm_import_path():
    from vetc.rssm.unified_rssm import UnifiedRSSM

    assert UnifiedRSSM.__name__ == "UnifiedRSSM"

