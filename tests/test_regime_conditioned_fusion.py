import torch

from vetc.fusion.regime_conditioned_fusion import RegimeConditionedFusion


def test_regime_conditioned_fusion_outputs_expected_shape_with_regime_token():
    fusion = RegimeConditionedFusion()
    z_ext = torch.randn(2, 128, 16, 16)
    z_int = torch.randn(2, 128)
    regime_token = torch.randn(2, 96)

    fused = fusion(z_ext, z_int, regime_token)

    assert fused.shape == (2, 256)
    assert torch.isfinite(fused).all().item()


def test_regime_conditioned_fusion_outputs_expected_shape_without_regime_token():
    fusion = RegimeConditionedFusion()
    z_ext = torch.randn(2, 128, 16, 16)
    z_int = torch.randn(2, 128)

    fused = fusion(z_ext, z_int, regime_token=None)

    assert fused.shape == (2, 256)
    assert torch.isfinite(fused).all().item()


def test_regime_conditioned_fusion_import_path():
    from vetc.fusion.regime_conditioned_fusion import RegimeConditionedFusion

    assert RegimeConditionedFusion.__name__ == "RegimeConditionedFusion"

