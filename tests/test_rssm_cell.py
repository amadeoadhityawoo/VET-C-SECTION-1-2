import torch

from vetc.rssm.rssm_cell import RSSMCell


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.randn(2, 256),
        torch.randn(2, 512),
        torch.randn(2, 64),
    )


def test_rssm_cell_outputs_expected_shapes_and_finite_tensors():
    cell = RSSMCell()
    fused, h_prev, z_prev = _inputs()

    output = cell(fused, h_prev, z_prev)

    assert output["h"].shape == (2, 512)
    assert output["z_prior"].shape == (2, 64)
    assert output["z_post"].shape == (2, 64)
    assert output["prior_mu"].shape == (2, 64)
    assert output["prior_logvar"].shape == (2, 64)
    assert output["post_mu"].shape == (2, 64)
    assert output["post_logvar"].shape == (2, 64)
    for tensor in output.values():
        assert torch.isfinite(tensor).all().item()


def test_rssm_cell_eval_mode_is_deterministic():
    cell = RSSMCell()
    cell.eval()
    fused, h_prev, z_prev = _inputs()

    with torch.no_grad():
        first = cell(fused, h_prev, z_prev)
        second = cell(fused, h_prev, z_prev)

    assert torch.equal(first["z_prior"], first["prior_mu"])
    assert torch.equal(first["z_post"], first["post_mu"])
    for key in first:
        assert torch.equal(first[key], second[key])

