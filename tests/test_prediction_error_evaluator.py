import torch

from vetc.prediction.prediction_error_evaluator import PredictionErrorEvaluator


def test_prediction_error_evaluator_fallback_outputs_expected_shapes():
    evaluator = PredictionErrorEvaluator()
    z_post = torch.randn(2, 64)
    z_prior = torch.randn(2, 64)

    output = evaluator(z_post, z_prior)

    assert output["surprise"].shape == (2, 1)
    assert output["kl_map"].shape == (2, 64)
    assert output["safe_sigma"].shape == (2, 64)
    for tensor in output.values():
        assert torch.isfinite(tensor).all().item()


def test_prediction_error_evaluator_gaussian_kl_outputs_expected_shapes():
    evaluator = PredictionErrorEvaluator()
    z_post = torch.randn(2, 64)
    z_prior = torch.randn(2, 64)
    prior_mu = torch.randn(2, 64)
    prior_logvar = torch.randn(2, 64).clamp(-2.0, 1.0)
    post_mu = torch.randn(2, 64)
    post_logvar = torch.randn(2, 64).clamp(-2.0, 1.0)

    output = evaluator(
        z_post,
        z_prior,
        prior_mu=prior_mu,
        prior_logvar=prior_logvar,
        post_mu=post_mu,
        post_logvar=post_logvar,
    )

    assert output["surprise"].shape == (2, 1)
    assert output["kl_map"].shape == (2, 64)
    assert output["safe_sigma"].shape == (2, 64)
    for tensor in output.values():
        assert torch.isfinite(tensor).all().item()
    assert (output["kl_map"] >= -1e-6).float().mean().item() > 0.99


def test_prediction_error_evaluator_import_path():
    from vetc.prediction.prediction_error_evaluator import PredictionErrorEvaluator

    assert PredictionErrorEvaluator.__name__ == "PredictionErrorEvaluator"

