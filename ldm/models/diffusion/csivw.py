from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch


def compute_csivw_gradient(
    losses: Sequence[torch.Tensor],
    optimize_tensor: torch.Tensor,
    state: Optional[Dict[str, torch.Tensor]] = None,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.25,
    eta: float = 0.25,
    ema_decay: float = 0.9,
    weight_temperature: float = 1.0,
    progress_kappa: float = 6.0,
    grad_clip: Optional[float] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Compute a consensus-stability inverse-variance weighted ensemble gradient.

    The function implements the CSIVW idea described in
    `docs/aege_new_weighting_proposal.md`:

    1. Treat each surrogate gradient as a noisy estimator of an unknown
       transferable target gradient.
    2. Estimate each surrogate's uncertainty with four proxy terms:
       - cross-model disagreement
       - temporal instability
       - gradient magnitude outlierness
       - loss progress failure
    3. Fuse the surrogate gradients with tempered inverse-variance weights.

    Args:
        losses:
            Scalar losses, one per surrogate model. Each loss must retain the
            autograd graph back to ``optimize_tensor``.
        optimize_tensor:
            Tensor with ``requires_grad=True``. In the current AdvDiffVLM code
            this is the latent variable ``img_n`` inside the adversarial DDIM
            loop.
        state:
            A persistent state dict returned by the previous call. Reuse the
            same dict across DDIM timesteps to enable temporal instability and
            EMA variance estimation.
        alpha, beta, gamma, eta:
            Coefficients for disagreement, temporal instability, magnitude
            outlierness and progress failure.
        ema_decay:
            Exponential moving average decay used for the uncertainty estimate.
        weight_temperature:
            Temperature applied to inverse-variance weighting.
            - 1.0 gives standard inverse-variance weighting.
            - >1.0 makes the weights flatter.
            - <1.0 makes the weights sharper.
        progress_kappa:
            Controls the slope of the progress sigmoid.
        grad_clip:
            Optional elementwise clipping threshold applied to each per-model
            gradient before aggregation.
        eps:
            Numerical stability constant.

    Returns:
        aggregated_gradient:
            The weighted surrogate gradient with the same shape as
            ``optimize_tensor``.
        new_state:
            State dict to feed into the next call.
        stats:
            Detached diagnostic tensors that are convenient for logging.
    """
    if len(losses) == 0:
        raise ValueError("`losses` must contain at least one surrogate loss.")
    if not optimize_tensor.requires_grad:
        raise ValueError("`optimize_tensor` must have requires_grad=True.")

    num_models = len(losses)
    device = optimize_tensor.device
    dtype = optimize_tensor.dtype

    detached_losses = []
    per_model_grads = []
    for idx, loss in enumerate(losses):
        scalar_loss = loss if loss.ndim == 0 else loss.mean()
        detached_losses.append(scalar_loss.detach().to(device=device, dtype=dtype))
        grad = torch.autograd.grad(
            scalar_loss,
            optimize_tensor,
            retain_graph=idx < num_models - 1,
            create_graph=False,
            allow_unused=False,
        )[0].detach()
        if grad_clip is not None:
            grad = torch.clamp(grad, min=-grad_clip, max=grad_clip)
        per_model_grads.append(grad)

    loss_values = torch.stack(detached_losses, dim=0)
    flat_grads = torch.stack([grad.reshape(-1) for grad in per_model_grads], dim=0)

    grad_norms = flat_grads.norm(dim=1).clamp_min(eps)
    unit_grads = flat_grads / grad_norms.unsqueeze(1)

    # 1) Consensus disagreement term D_i = 1 - mean_j max(0, cos(u_i, u_j))
    if num_models == 1:
        disagreement = torch.zeros(1, device=device, dtype=dtype)
    else:
        cosine_matrix = torch.matmul(unit_grads, unit_grads.t()).clamp(-1.0, 1.0)
        positive_cosine = torch.clamp(cosine_matrix, min=0.0)
        consensus = (positive_cosine.sum(dim=1) - positive_cosine.diagonal()) / max(num_models - 1, 1)
        disagreement = 1.0 - consensus

    # 2) Temporal instability term T_i = 1 - max(0, cos(u_i^t, u_i^{t-1}))
    temporal_instability = torch.zeros(num_models, device=device, dtype=dtype)
    if state is not None and "prev_unit_grads" in state:
        prev_unit_grads = state["prev_unit_grads"].to(device=device, dtype=dtype)
        if prev_unit_grads.shape == unit_grads.shape:
            temporal_cosine = (unit_grads * prev_unit_grads).sum(dim=1).clamp(min=0.0, max=1.0)
            temporal_instability = 1.0 - temporal_cosine

    # 3) Magnitude outlier term M_i = |log ||g_i|| - median_j log ||g_j|||
    log_norms = torch.log(grad_norms + eps)
    norm_median = log_norms.median()
    magnitude_outlier = (log_norms - norm_median).abs()

    # 4) Progress failure term F_i = 1 - sigmoid(kappa * deltaL_i)
    progress_failure = torch.zeros(num_models, device=device, dtype=dtype)
    if state is not None and "prev_loss_values" in state:
        prev_loss_values = state["prev_loss_values"].to(device=device, dtype=dtype)
        if prev_loss_values.shape == loss_values.shape:
            delta_loss = (loss_values - prev_loss_values) / prev_loss_values.abs().clamp_min(eps)
            progress = torch.sigmoid(progress_kappa * delta_loss)
            progress_failure = 1.0 - progress

    sigma2_hat = (
        alpha * disagreement
        + beta * temporal_instability
        + gamma * magnitude_outlier
        + eta * progress_failure
        + eps
    )

    if state is not None and "ema_sigma2" in state:
        prev_sigma2 = state["ema_sigma2"].to(device=device, dtype=dtype)
        if prev_sigma2.shape == sigma2_hat.shape:
            sigma2 = ema_decay * prev_sigma2 + (1.0 - ema_decay) * sigma2_hat
        else:
            sigma2 = sigma2_hat
    else:
        sigma2 = sigma2_hat

    temperature = max(float(weight_temperature), eps)
    tempered_inverse_variance = sigma2.clamp_min(eps).pow(-1.0 / temperature)
    weights = tempered_inverse_variance / tempered_inverse_variance.sum().clamp_min(eps)

    aggregated_gradient = torch.zeros_like(optimize_tensor)
    for weight, grad in zip(weights, per_model_grads):
        aggregated_gradient = aggregated_gradient + weight * grad

    new_state = {
        "prev_unit_grads": unit_grads.detach(),
        "prev_loss_values": loss_values.detach(),
        "ema_sigma2": sigma2.detach(),
    }
    stats = {
        "weights": weights.detach(),
        "sigma2_hat": sigma2_hat.detach(),
        "sigma2": sigma2.detach(),
        "consensus_disagreement": disagreement.detach(),
        "temporal_instability": temporal_instability.detach(),
        "magnitude_outlier": magnitude_outlier.detach(),
        "progress_failure": progress_failure.detach(),
        "grad_norms": grad_norms.detach(),
        "loss_values": loss_values.detach(),
    }
    return aggregated_gradient, new_state, stats


__all__ = ["compute_csivw_gradient"]
