from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def build_clipseg_predictor(
    model_name: str = "CIDAS/clipseg-rd64-refined",
    device: Optional[torch.device | str] = None,
) -> Callable[[torch.Tensor, Sequence[str]], torch.Tensor]:
    """
    Build a CLIPSeg predictor callable.

    The returned callable accepts:
    - image: a tensor in shape [1, 3, H, W], range [0, 1]
    - prompts: a sequence of keyword phrases

    and returns CLIPSeg logits in shape [num_prompts, H', W'].
    """
    from torchvision.transforms.functional import to_pil_image
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPSegProcessor.from_pretrained(model_name)
    model = CLIPSegForImageSegmentation.from_pretrained(model_name).to(device)
    model.eval()

    @torch.no_grad()
    def predictor(image: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("CLIPSeg predictor expects `image` with shape [1, 3, H, W].")
        if len(prompts) == 0:
            raise ValueError("`prompts` must contain at least one keyword phrase.")

        image = image.detach().float().cpu().clamp(0.0, 1.0).squeeze(0)
        pil_image = to_pil_image(image)
        inputs = processor(
            text=list(prompts),
            images=[pil_image] * len(prompts),
            padding=True,
            return_tensors="pt",
        )
        inputs = {name: value.to(device) for name, value in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        if logits.ndim == 4 and logits.shape[1] == 1:
            logits = logits[:, 0]
        return logits.detach()

    return predictor


def normalize_clipseg_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Apply the document's normalization:

        N(Z) = sigmoid(Z / tau_seg) / (max(sigmoid(Z / tau_seg)) + eps)
    """
    if logits.ndim != 3:
        raise ValueError("`logits` must have shape [num_prompts, H, W].")
    scaled = torch.sigmoid(logits / max(float(temperature), eps))
    max_values = scaled.amax(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return scaled / max_values


def compute_keyword_region_prior(
    target_image: torch.Tensor,
    prompts: Sequence[str],
    prompt_weights: Optional[Sequence[float]],
    predictor: Callable[[torch.Tensor, Sequence[str]], torch.Tensor],
    *,
    seg_temperature: float = 1.0,
    output_size: Optional[Tuple[int, int]] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute the static target prior region map R from the target image.
    """
    weights = _normalize_prompt_weights(prompts, prompt_weights, target_image.device, target_image.dtype)
    logits = predictor(target_image, prompts).to(device=target_image.device, dtype=target_image.dtype)
    per_prompt_maps = normalize_clipseg_logits(logits, temperature=seg_temperature, eps=eps)
    if output_size is not None:
        per_prompt_maps = _resize_prompt_maps(per_prompt_maps, output_size)
    prior_region = torch.sum(weights[:, None, None] * per_prompt_maps, dim=0)
    return prior_region, {
        "prior_region": prior_region.detach(),
        "prior_per_prompt": per_prompt_maps.detach(),
        "prompt_weights": weights.detach(),
    }


def compute_kgrc_mask(
    current_image: torch.Tensor,
    prompts: Sequence[str],
    prompt_weights: Optional[Sequence[float]],
    predictor: Callable[[torch.Tensor, Sequence[str]], torch.Tensor],
    *,
    state: Optional[Dict[str, torch.Tensor]] = None,
    target_image: Optional[torch.Tensor] = None,
    prior_region: Optional[torch.Tensor] = None,
    attack_step_idx: int = 0,
    total_attack_steps: int = 1,
    seg_temperature: float = 1.0,
    ema_decay: float = 0.8,
    lambda_min: float = 0.35,
    lambda_max: float = 0.85,
    lambda_beta: float = 2.0,
    fusion_mode: str = "poe",
    output_size: Optional[Tuple[int, int]] = None,
    output_channels: Optional[int] = None,
    gate_power: float = 1.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Compute the dynamic keyword-grounded region constraint mask M_t.

    This function implements the full pipeline from the innovation note:
    1. Build or reuse the target prior region R.
    2. Predict the current response map A^(t) on the denoised image estimate.
    3. Smooth the response with EMA.
    4. Fuse prior and observation with either:
       - `poe`: weighted product-of-experts fusion in logit space
       - `linear`: convex combination
    5. Optionally resize and broadcast to latent shape for gradient gating.
    """
    device = current_image.device
    dtype = current_image.dtype
    weights = _normalize_prompt_weights(prompts, prompt_weights, device, dtype)

    if state is None:
        state = {}

    if prior_region is None:
        if "prior_region" in state:
            prior_region = state["prior_region"].to(device=device, dtype=dtype)
        else:
            if target_image is None:
                raise ValueError("`target_image` is required to initialize the keyword region prior.")
            prior_region, prior_stats = compute_keyword_region_prior(
                target_image,
                prompts,
                prompt_weights,
                predictor,
                seg_temperature=seg_temperature,
                output_size=output_size or current_image.shape[-2:],
                eps=eps,
            )
            prior_region = prior_region.to(device=device, dtype=dtype)
            state["prior_region"] = prior_region.detach()
            state["prior_per_prompt"] = prior_stats["prior_per_prompt"].to(device=device, dtype=dtype)

    current_logits = predictor(current_image, prompts).to(device=device, dtype=dtype)
    current_per_prompt = normalize_clipseg_logits(current_logits, temperature=seg_temperature, eps=eps)
    if prior_region.shape[-2:] != current_per_prompt.shape[-2:]:
        current_per_prompt = _resize_prompt_maps(current_per_prompt, prior_region.shape[-2:])
    current_response = torch.sum(weights[:, None, None] * current_per_prompt, dim=0)

    if "ema_response" in state:
        previous_ema = state["ema_response"].to(device=device, dtype=dtype)
        if previous_ema.shape == current_response.shape:
            ema_response = ema_decay * previous_ema + (1.0 - ema_decay) * current_response
        else:
            ema_response = current_response
    else:
        ema_response = current_response

    lambda_t = compute_lambda_schedule(
        attack_step_idx=attack_step_idx,
        total_attack_steps=total_attack_steps,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        lambda_beta=lambda_beta,
    )
    fused_mask = fuse_region_maps(
        prior_region,
        ema_response,
        lambda_t=lambda_t,
        mode=fusion_mode,
        eps=eps,
    )

    output_mask = fused_mask
    if output_size is not None and fused_mask.shape[-2:] != output_size:
        output_mask = _resize_single_map(fused_mask, output_size)
    output_mask = output_mask.clamp(0.0, 1.0)
    if gate_power != 1.0:
        output_mask = output_mask.pow(gate_power)

    if output_channels is not None:
        output_mask = output_mask.unsqueeze(0).unsqueeze(0)
        output_mask = output_mask.expand(1, output_channels, *output_mask.shape[-2:])

    new_state = {
        "prior_region": prior_region.detach(),
        "ema_response": ema_response.detach(),
        "prior_per_prompt": state.get("prior_per_prompt"),
    }
    stats = {
        "prior_region": prior_region.detach(),
        "current_response": current_response.detach(),
        "ema_response": ema_response.detach(),
        "lambda_t": torch.tensor(lambda_t, device=device, dtype=dtype),
        "fused_mask": fused_mask.detach(),
        "output_mask": output_mask.detach(),
    }
    return output_mask, new_state, stats


def compute_lambda_schedule(
    attack_step_idx: int,
    total_attack_steps: int,
    *,
    lambda_min: float = 0.35,
    lambda_max: float = 0.85,
    lambda_beta: float = 2.0,
) -> float:
    total_attack_steps = max(int(total_attack_steps), 1)
    progress = float(attack_step_idx) / float(max(total_attack_steps - 1, 1))
    lambda_t = lambda_min + (lambda_max - lambda_min) * math.exp(-lambda_beta * progress)
    return float(max(min(lambda_t, lambda_max), lambda_min))


def fuse_region_maps(
    prior_region: torch.Tensor,
    observation_region: torch.Tensor,
    *,
    lambda_t: float,
    mode: str = "poe",
    eps: float = 1e-6,
) -> torch.Tensor:
    if prior_region.shape != observation_region.shape:
        raise ValueError("`prior_region` and `observation_region` must share the same shape.")

    if mode == "linear":
        return lambda_t * prior_region + (1.0 - lambda_t) * observation_region
    if mode != "poe":
        raise ValueError(f"Unsupported fusion mode: {mode}")

    prior = prior_region.clamp(eps, 1.0 - eps)
    obs = observation_region.clamp(eps, 1.0 - eps)
    logits = lambda_t * _safe_logit(prior, eps=eps) + (1.0 - lambda_t) * _safe_logit(obs, eps=eps)
    return torch.sigmoid(logits)


def _normalize_prompt_weights(
    prompts: Sequence[str],
    prompt_weights: Optional[Sequence[float]],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if len(prompts) == 0:
        raise ValueError("`prompts` must contain at least one keyword phrase.")

    if prompt_weights is None:
        weights = torch.full((len(prompts),), 1.0 / len(prompts), device=device, dtype=dtype)
    else:
        if len(prompt_weights) != len(prompts):
            raise ValueError("`prompt_weights` must have the same length as `prompts`.")
        weights = torch.tensor(prompt_weights, device=device, dtype=dtype)
        weights = weights / weights.sum().clamp_min(1e-6)
    return weights


def _safe_logit(probabilities: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probabilities = probabilities.clamp(eps, 1.0 - eps)
    return torch.log(probabilities) - torch.log1p(-probabilities)


def _resize_prompt_maps(prompt_maps: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
    resized = F.interpolate(
        prompt_maps.unsqueeze(1),
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )
    return resized[:, 0]


def _resize_single_map(mask: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
    resized = F.interpolate(
        mask.unsqueeze(0).unsqueeze(0),
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )
    return resized[0, 0]


__all__ = [
    "build_clipseg_predictor",
    "normalize_clipseg_logits",
    "compute_keyword_region_prior",
    "compute_kgrc_mask",
    "compute_lambda_schedule",
    "fuse_region_maps",
]
