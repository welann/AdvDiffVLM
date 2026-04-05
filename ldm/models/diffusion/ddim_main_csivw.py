"""CSIVW-integrated DDIM sampler.

This module re-exports the original `ddim_main` symbols and overrides only the
`DDIMSampler.ddim_sampling` method so it can serve as a drop-in replacement.
"""

import numpy as np
import torch
from tqdm import tqdm

from ldm.models.diffusion import ddim_main as ddim_main_base
from ldm.models.diffusion.csivw import compute_csivw_gradient
from ldm.models.diffusion.ddim_main import *  # noqa: F401,F403


class DDIMSampler(ddim_main_base.DDIMSampler):
    @torch.no_grad()
    def ddim_sampling(
        self,
        cond,
        shape,
        x_T=None,
        ddim_use_original_steps=False,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        cam=None,
        x0=None,
        img_callback=None,
        log_every_t=100,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        label=None,
        tgt_image_features_list=None,
        org_image_features_list=None,
        K=10,
        s=0.75,
        a=0.5,
    ):
        device = self.model.betas.device
        b = shape[0]
        base_x0 = x_T if x_T is not None else x0
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            t = torch.full((1,), 201, device=device, dtype=torch.long)
            img = self.model.q_sample(x_T, t, noise=torch.randn_like(x_T.float()))

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {"x_inter": [img], "pred_x0": [img]}
        time_range = list(reversed(range(0, timesteps))) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]

        num_models = len(self.models) if self.models is not None else 0
        if num_models == 0:
            raise ValueError("CSIVW DDIM sampler requires `self.models` to contain at least one surrogate model.")
        if tgt_image_features_list is None:
            raise ValueError("CSIVW DDIM sampler requires `tgt_image_features_list`.")
        if len(tgt_image_features_list) != num_models:
            raise ValueError(
                f"Mismatched surrogate count: len(self.models)={num_models}, "
                f"len(tgt_image_features_list)={len(tgt_image_features_list)}."
            )

        pri_img = img.detach().requires_grad_(True)

        for k in range(K):
            img = pri_img.detach().requires_grad_(True)

            print(f"Running Adversarial Sampling at {k} step")
            print(f"Running DDIM Sampling with {total_steps} timesteps")

            iterator = tqdm(time_range, desc="DDIM Sampler", total=total_steps)

            costs = torch.zeros((total_steps, num_models), dtype=torch.float32)
            weights = torch.zeros((total_steps, num_models), dtype=torch.float32)
            csivw_state = None
            attack_step_idx = 0

            for i, step in enumerate(iterator):
                index = total_steps - i - 1
                if index > total_steps * 0.2:
                    continue

                ts = torch.full((b,), step, device=device, dtype=torch.long)
                mask = None
                if cam is not None:
                    mask = cam.clone().to(device)
                    mask = torch.clamp(mask, 0.045, 1.0)

                    cam_new = torch.clamp(cam, 0.3, 0.7)
                    prob_matrix = cam_new.detach().cpu().numpy()
                    prob_matrix /= prob_matrix.sum()

                    x, y = ddim_main_base.sample_coordinates(prob_matrix)
                    x_left, x_right = max(0, x - 4), min(64, x + 4)
                    y_left, y_right = max(0, y - 4), min(64, y + 4)
                    mask[x_left:x_right, y_left:y_right] = 0

                if mask is not None:
                    if base_x0 is None:
                        raise ValueError("Masked CSIVW DDIM sampling requires `x_T` or `x0` as the base image latent.")
                    img_orig = self.model.q_sample(base_x0, ts)
                    img = img_orig * mask + (1.0 - mask) * img

                outs = self.p_sample_ddim(
                    img,
                    cond,
                    ts,
                    index=index,
                    use_original_steps=ddim_use_original_steps,
                    quantize_denoised=quantize_denoised,
                    temperature=temperature,
                    noise_dropout=noise_dropout,
                    score_corrector=score_corrector,
                    corrector_kwargs=corrector_kwargs,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=unconditional_conditioning,
                )
                img, pred_x0 = outs

                if index <= total_steps * 0.2:
                    for _ in range(1):
                        with torch.enable_grad():
                            img_n = img.detach().requires_grad_(True)
                            img_transformed = self.model.differentiable_decode_first_stage(img_n)
                            img_transformed = torch.clamp((img_transformed + 1.0) / 2.0, min=0.0, max=1.0)
                            img_transformed = self.preprocess(img_transformed)

                            adv_image_feature_list = []
                            for model in self.models:
                                adv_image_features = model.encode_image(img_transformed)
                                adv_image_features = adv_image_features / adv_image_features.norm(
                                    dim=1, keepdim=True
                                )
                                adv_image_feature_list.append(adv_image_features)

                            per_model_losses = []
                            for model_i, (pred_i, target_i) in enumerate(
                                zip(adv_image_feature_list, tgt_image_features_list)
                            ):
                                crit1 = torch.mean(torch.sum(pred_i * target_i, dim=1))
                                per_model_losses.append(crit1)
                                costs[attack_step_idx, model_i] = crit1.detach().item()

                            gradient, csivw_state, csivw_stats = compute_csivw_gradient(
                                per_model_losses,
                                img_n,
                                state=csivw_state,
                                alpha=1.0,
                                beta=1.0,
                                gamma=0.25,
                                eta=0.25,
                                ema_decay=0.9,
                                weight_temperature=1.0,
                                progress_kappa=6.0,
                                grad_clip=0.0025,
                            )

                        weights[attack_step_idx, :num_models] = csivw_stats["weights"].detach().cpu()
                        img = img + s * gradient
                    attack_step_idx += 1

                if callback:
                    callback(i)
                if img_callback:
                    img_callback(pred_x0, i)

                if index % log_every_t == 0 or index == total_steps - 1:
                    intermediates["x_inter"].append(img)
                    intermediates["pred_x0"].append(pred_x0)

            self.csivw_logs = {
                "costs": costs[:attack_step_idx].clone(),
                "weights": weights[:attack_step_idx].clone(),
            }

        return img, intermediates


__all__ = [name for name in globals() if not name.startswith("_")]
