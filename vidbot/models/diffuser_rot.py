from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import vidbot.diffuser_utils.tensor_utils as TensorUtils
import vidbot.diffuser_utils.dataset_utils as DataUtils
from vidbot.models.helpers import cosine_beta_schedule, extract, default
from vidbot.models.feature_extractors import MultiScaleImageFeatureExtractor
from vidbot.models.temporal import TemporalMapUnet


class DiffuserRotationModel(nn.Module):
    def __init__(
        self,
        n_timesteps=100,
        loss_type="l2",
        horizon=40,
        observation_dim=6,
        output_dim=6,
        base_dim=32,
        dim_mults=[2, 4, 8],
        vlm_feature_dim=512,
        object_cond_feature_dim=256,
        action_cond_feature_dim=256,
        diffuser_model_arch="TemporalMapUnet",
        diffuser_use_preceiver=False,
        object_image_shape=[256, 256],
        cond_fill_value=-1.0,
        predict_epsilons=False,
        supervise_epsilons=False,
        **kwargs,
    ):
        super(DiffuserRotationModel, self).__init__()
        self.n_timesteps = int(n_timesteps)
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.object_cond_feature_dim = object_cond_feature_dim
        self.action_cond_feature_dim = action_cond_feature_dim
        self.cond_feature_dim = object_cond_feature_dim + action_cond_feature_dim
        self.output_dim = output_dim
        self.base_dim = base_dim
        self.dim_mults = dim_mults
        self.object_image_shape = object_image_shape
        self.cond_fill_value = cond_fill_value
        self.predict_epsilons = predict_epsilons
        self.supervise_epsilons = supervise_epsilons

        ## diffuser architecture
        self.register_diffusion_params()
        if diffuser_model_arch == "TemporalMapUnet":
            self.transition_in_dim = self.observation_dim
            # According to Trace, the model directly predicts the start state from the noisy state
            self.model = TemporalMapUnet(
                horizon=self.horizon,
                transition_dim=self.transition_in_dim,
                cond_dim=self.cond_feature_dim,
                output_dim=self.output_dim,
                dim=self.base_dim,
                dim_mults=self.dim_mults,
                use_preceiver=diffuser_use_preceiver,
            )
        else:
            print("unknown diffuser_model_arch:", diffuser_model_arch)
            raise

        self.object_feature_extractor = MultiScaleImageFeatureExtractor(
            embedding_dim=object_cond_feature_dim
        )

        self.action_feature_extractor = nn.Sequential(
            nn.TransformerEncoderLayer(
                d_model=vlm_feature_dim,
                nhead=4,
                dim_feedforward=512,
                batch_first=True,
            ),
            nn.Linear(vlm_feature_dim, action_cond_feature_dim),
        )

        self.loss_type = loss_type
        self.current_guidance = None

    def register_diffusion_params(self):
        betas = cosine_beta_schedule(self.n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

        # calculations for class-free guidance
        self.sqrt_alphas_over_one_minus_alphas_cumprod = torch.sqrt(
            alphas_cumprod / (1.0 - alphas_cumprod)
        )
        self.sqrt_recip_one_minus_alphas_cumprod = 1.0 / torch.sqrt(1.0 - alphas_cumprod)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def set_guidance(self, guidance):
        """
        Instantiates test-time guidance functions using the list of configs (dicts) passed in.
        """
        # FIXME: change the implementation of guidance later on
        self.current_guidance = guidance

    def scale_trajectory(self, traj, min_bound, max_bound):
        """
        - traj: B x H x 3
        """

        return traj

    def descale_trajectory(self, traj, min_bound, max_bound):
        """
        - traj: B x N x H x 3
        """
        return traj

    def forward(
        self,
        data_batch,
        num_samp=1,
        return_diffusion=False,
        return_guidance_losses=False,
        class_free_guide_w=0.0,
        apply_guidance=True,
        guide_clean=False,
    ):
        use_class_free_guide = class_free_guide_w != 0.0
        aux_info = self.get_aux_info(data_batch, include_class_free_cond=use_class_free_guide)
        cond_samp_out = self.conditional_sample(
            data_batch,
            horizon=None,
            aux_info=aux_info,
            return_diffusion=return_diffusion,
            return_guidance_losses=return_guidance_losses,
            num_samp=num_samp,
            class_free_guide_w=class_free_guide_w,
            apply_guidance=apply_guidance,
            guide_clean=guide_clean,
        )
        traj_scaled = cond_samp_out["pred_trajectory"]

        gt_traj_min_bound = data_batch["gt_traj_min_bound"]
        gt_traj_max_bound = data_batch["gt_traj_max_bound"]

        traj_r6d_res = self.descale_trajectory(traj_scaled, gt_traj_min_bound, gt_traj_max_bound)

        outputs = {}
        traj_rot_res = DataUtils.r6d_to_rotation_matrix(traj_r6d_res)  # [B, N, H, 3, 3]
        traj_rot_res_init = traj_rot_res[:, :, :1]  # [B, N, 3, 3]
        traj_rot_res_init = traj_rot_res_init.repeat(
            1, 1, traj_rot_res.shape[2], 1, 1
        )  # [B, N, H, 3, 3]
        traj_rot_res = torch.matmul(
            traj_rot_res, traj_rot_res_init.transpose(-1, -2)
        )  # [B, N, H, 3, 3]

        rot_init = data_batch["gt_rot_init"][:, None, None]  # [B, 1, 1, 3, 3]
        rot_init = rot_init.repeat(1, traj_rot_res.shape[1], traj_rot_res.shape[2], 1, 1)
        traj_rot = torch.matmul(traj_rot_res, rot_init)  # [B, N, H, 3, 3]
        outputs["predictions_rot"] = traj_rot

        return outputs

    def compute_losses(self, data_batch):
        aux_info = self.get_aux_info(data_batch, training=True)
        traj = data_batch["gt_r6d_res_trajectory"]

        gt_traj_min_bound = data_batch["gt_traj_min_bound"]
        gt_traj_max_bound = data_batch["gt_traj_max_bound"]

        x = self.scale_trajectory(traj, gt_traj_min_bound, gt_traj_max_bound)
        diffusion_loss = self.loss(x, aux_info=aux_info)
        losses = OrderedDict(
            diffusion_loss=diffusion_loss,
        )
        return losses

    def get_loss_weights(self, action_weight, discount):
        dim_weights = torch.ones(self.observation_dim, dtype=torch.float32)
        return dim_weights

    def get_aux_info(self, data_batch, include_class_free_cond=False, training=False):
        if training:
            object_color = data_batch["object_color_aug"]
        else:
            object_color = data_batch["object_color"]

        intrinsics = data_batch["intrinsics"]

        object_feature = self.object_feature_extractor(object_color)
        action_feature_vlm = data_batch["action_feature"]
        action_feature = self.action_feature_extractor(action_feature_vlm)
        cond_feature = torch.cat([object_feature, action_feature], dim=-1)

        gt_traj_min_bound = data_batch["gt_traj_min_bound"]
        gt_traj_max_bound = data_batch["gt_traj_max_bound"]
        aux_info = {
            "cond_feat": cond_feature,
            "intrinsics": intrinsics,
            "gt_traj_min_bound": gt_traj_min_bound,
            "gt_traj_max_bound": gt_traj_max_bound,
        }

        if include_class_free_cond:
            object_color_non_cond = torch.ones_like(object_color) * self.cond_fill_value
            object_feature_non_cond = self.object_feature_extractor(object_color_non_cond)
            action_feature_non_cond = data_batch["action_feature_null"]
            action_feature_non_cond = self.action_feature_extractor(action_feature_non_cond)
            non_cond_feature = torch.cat(
                [object_feature_non_cond, action_feature_non_cond],
                dim=-1,
            )
            aux_info["non_cond_feat"] = non_cond_feature

        return aux_info

    # ------------------------------------------ sampling ------------------------------------------#
    def predict_start_from_noise(self, x_t, t, noise, force_noise=False):
        if force_noise:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def predict_noise_from_start(self, x_t, t, x_start):
        return (
            extract(self.sqrt_recip_one_minus_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t
            - extract(
                self.sqrt_alphas_over_one_minus_alphas_cumprod.to(x_t.device),
                t,
                x_t.shape,
            )
            * x_start
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def guidance(self, x, t, data_batch, aux_info, num_samp=1, return_grad_of=None):
        """
        estimate the gradient of rule reward w.r.t. the input trajectory
        Input:
            x: [batch_size*num_samp, time_steps, feature_dim].  scaled input trajectory.
            data_batch: additional info.
            aux_info: additional info.
            return_grad_of: which variable to take gradient of guidance loss wrt, if not given,
                            takes wrt the input x.
        """
        assert self.current_guidance is not None, "Must instantiate guidance object before calling"
        bsize = int(x.size(0) / num_samp)
        horizon = x.size(1)
        with torch.enable_grad():
            # compute losses and gradient
            x_loss = x.reshape((bsize, num_samp, horizon, -1))
            tot_loss, per_losses = self.current_guidance.compute_guidance_loss(
                x_loss, t, data_batch
            )
            # print(tot_loss)
            tot_loss.backward()
            guide_grad = x.grad if return_grad_of is None else return_grad_of.grad

            return guide_grad, per_losses

    def p_mean_variance(self, x, t, aux_info={}, class_free_guide_w=0.0):
        model_prediction = self.model(x, aux_info["cond_feat"], t)

        if class_free_guide_w != 0.0:
            x_non_cond_inp = x.clone()

            # model predicts noise from that brings t to t-1
            model_non_cond_prediction = self.model(x_non_cond_inp, aux_info["non_cond_feat"], t)

            if not self.predict_epsilons:
                # ... and combine to get actual model prediction (in noise space as in original paper)
                model_pred_noise = self.predict_noise_from_start(
                    x_t=x, t=t, x_start=model_prediction
                )  # noise 1
                model_non_cond_pred_noise = self.predict_noise_from_start(
                    x_t=x, t=t, x_start=model_non_cond_prediction
                )  # noise 2
                class_free_guide_noise = (
                    1 + class_free_guide_w
                ) * model_pred_noise - class_free_guide_w * model_non_cond_pred_noise  # compose noise
                model_prediction = self.predict_start_from_noise(
                    x_t=x, t=t, noise=class_free_guide_noise, force_noise=True
                )  # get actual model prediction by sampling back
                # x_recon = self.predict_start_from_noise(
                #     x, t=t, noise=model_prediction, force_noise=False
                # )  # x_recon = model_prediction
            else:
                model_pred_noise = model_prediction
                model_non_cond_pred_noise = model_non_cond_prediction
                class_free_guide_noise = (
                    1 + class_free_guide_w
                ) * model_pred_noise - class_free_guide_w * model_non_cond_pred_noise
                model_prediction = class_free_guide_noise
                # x_recon = self.predict_start_from_noise(
                #     x, t=t, noise=model_prediction, force_noise=True
                # )
        x_recon = self.predict_start_from_noise(
            x, t=t, noise=model_prediction, force_noise=self.predict_epsilons
        )  # x_recon = model_prediction
        # if self.predict_epsilons:
        x_recon.clamp_(-1, 1)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )  # q(x_{t-1} | x_t, x_0)
        return model_mean, posterior_variance, posterior_log_variance, (x_recon, x, t)

    @torch.no_grad()
    def p_sample(
        self,
        x,
        t,
        data_batch,
        aux_info={},
        num_samp=1,
        class_free_guide_w=0.0,
        apply_guidance=True,
        guide_clean=True,
        eval_final_guide_loss=False,
    ):
        # NOTE: guide_clean is usually True
        b, *_, _device = *x.shape, x.device
        with_func = torch.no_grad
        # print("===> Time {} X_in {}".format(t, x.mean()))
        if self.current_guidance is not None and apply_guidance and guide_clean:
            # will need to take grad wrt noisy input
            x = x.detach()
            x.requires_grad_()
            with_func = torch.enable_grad

        with with_func():
            # get prior mean and variance for next step => q(x_{t-1} | x_t, x_0)
            model_mean, _, model_log_variance, q_posterior_in = self.p_mean_variance(
                x=x, t=t, aux_info=aux_info, class_free_guide_w=class_free_guide_w
            )

        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        noise = torch.randn_like(model_mean)
        sigma = (0.5 * model_log_variance).exp()

        # compute guidance
        guide_losses = None
        guide_grad = torch.zeros_like(model_mean)
        if self.current_guidance is not None and apply_guidance:
            assert not self.predict_epsilons, "Guidance not implemented for epsilon prediction"
            if guide_clean:  # Return gradients of x_{t-1}
                # We want to guide the predicted clean traj from model, not the noisy one
                model_clean_pred = q_posterior_in[0]
                x_guidance = model_clean_pred
                return_grad_of = x
            else:  # Returerequires_grad gradients of x_0
                x_guidance = model_mean.clone().detach()
                return_grad_of = x_guidance
                x_guidance.requires_grad_()
            # TODO: Look into how gradient computation in guidance works, and implement our own guidance
            guide_grad, guide_losses = self.guidance(
                x_guidance,
                t,
                data_batch,
                aux_info,
                num_samp=num_samp,
                return_grad_of=return_grad_of,
            )

            # NOTE: empirally, scaling by the variance (sigma) seems to degrade results
            guide_grad = nonzero_mask * guide_grad  # * sigma

        noise = nonzero_mask * sigma * noise

        if self.current_guidance is not None and guide_clean:
            assert not self.predict_epsilons, "Guidance not implemented for epsilon prediction"
            # perturb clean trajectory
            guided_clean = (
                q_posterior_in[0] - guide_grad
            )  # x_0' = x_0 - grad (The use of guidance)
            # use the same noisy input again
            guided_x_t = q_posterior_in[1]  # x_{t}
            # re-compute next step distribution with guided clean & noisy trajectories => q(x_{t-1}|x_{t}, x_0')
            # And remember in the training process, we want to make the output of every diffusion step to be x_0
            model_mean, _, _ = self.q_posterior(
                x_start=guided_clean, x_t=guided_x_t, t=q_posterior_in[2]
            )
            # NOTE: variance is not dependent on x_start, so it won't change. Therefore, fine to use same noise.
            x_out = model_mean + noise
        else:
            x_out = model_mean - guide_grad + noise

        if self.current_guidance is not None and eval_final_guide_loss:
            assert not self.predict_epsilons, "Guidance not implemented for epsilon prediction"
            # eval guidance loss one last time for filtering if desired
            #       (even if not applied during sampling)
            _, guide_losses = self.guidance(
                x_out.clone().detach().requires_grad_(),
                t,
                data_batch,
                aux_info,
                num_samp=num_samp,
            )
        return x_out, guide_losses

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        data_batch,
        num_samp,
        aux_info={},
        return_diffusion=False,
        return_guidance_losses=False,
        class_free_guide_w=0.0,
        apply_guidance=True,
        guide_clean=False,
    ):
        device = self.betas.device

        batch_size = shape[0]
        # sample from base distribution
        x = torch.randn(shape, device=device)  # (B, num_samp, horizon, transition)

        x = TensorUtils.join_dimensions(
            x, begin_axis=0, end_axis=2
        )  # B*num_samp, horizon, transition
        aux_info = TensorUtils.repeat_by_expand_at(aux_info, repeats=num_samp, dim=0)

        if self.current_guidance is not None and not apply_guidance:
            print(
                "DIFFUSER: Note, not using guidance during sampling, only evaluating guidance loss at very end..."
            )

        if return_diffusion:
            diffusion = [x]

        stride = 1  # NOTE: different from training time if > 1
        steps = [i for i in reversed(range(0, self.n_timesteps, stride))]

        for i in steps:
            timesteps = torch.full((batch_size * num_samp,), i, device=device, dtype=torch.long)
            x, guide_losses = self.p_sample(
                x,
                timesteps,
                data_batch,
                aux_info=aux_info,
                num_samp=num_samp,
                class_free_guide_w=class_free_guide_w,
                apply_guidance=apply_guidance,
                guide_clean=guide_clean,
                eval_final_guide_loss=(i == steps[-1]),
            )
            if return_diffusion:
                diffusion.append(x)

        if guide_losses is not None:
            print("===== GUIDANCE LOSSES ======")
            for k, v in guide_losses.items():
                print("%s: %.012f" % (k, np.nanmean(v.cpu())))

        x = TensorUtils.reshape_dimensions(
            x, begin_axis=0, end_axis=1, target_dims=(batch_size, num_samp)
        )

        out_dict = {"pred_trajectory": x}
        if return_guidance_losses:
            out_dict["guide_losses"] = guide_losses

        if return_diffusion:
            diffusion = [
                TensorUtils.reshape_dimensions(
                    cur_diff,
                    begin_axis=0,
                    end_axis=1,
                    target_dims=(batch_size, num_samp),
                )
                for cur_diff in diffusion
            ]
            out_dict["diffusion"] = torch.stack(diffusion, dim=3)
        return out_dict

    @torch.no_grad()
    def conditional_sample(
        self, data_batch, horizon=None, num_samp=1, class_free_guide_w=0.0, **kwargs
    ):
        try:
            batch_size = data_batch["color"].shape[0]
        except Exception:
            batch_size = data_batch["color_aug"].shape[0]
        horizon = horizon or self.horizon
        shape = (batch_size, num_samp, horizon, self.observation_dim)
        return self.p_sample_loop(
            shape,
            data_batch,
            num_samp=num_samp,
            class_free_guide_w=class_free_guide_w,
            **kwargs,
        )

    # ------------------------------------------ training ------------------------------------------#
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    def p_losses(self, x_start_init, t, aux_info={}, noise=None):
        # noise_init = torch.randn_like(x_start_init)
        noise = default(noise, lambda: torch.randn_like(x_start_init))
        # Forward process to get x_t
        x_start = x_start_init
        # noise = noise_init
        x_noisy = self.q_sample(x_start, t, noise)
        t_inp = t

        # Reverse process to predict x_start from x_t
        # The input to the noise model includes the map features

        x_noisy_inp = x_noisy
        model_prediction = self.model(x_noisy_inp, aux_info["cond_feat"], t_inp)
        x_recon = self.predict_start_from_noise(
            x_noisy, t_inp, model_prediction, force_noise=self.predict_epsilons
        )  # x_recon = noise

        if not self.predict_epsilons:
            noise_pred = self.predict_noise_from_start(
                x_noisy, t_inp, x_recon
            )  # noise_pred = x_recon

        else:
            x_recon = self.predict_start_from_noise(
                x_noisy, t_inp, model_prediction, force_noise=True
            )
            noise_pred = model_prediction

        if self.supervise_epsilons:
            assert self.predict_epsilons
            loss = self.loss_fn(noise_pred, noise)
        else:
            assert not self.predict_epsilons
            loss = self.loss_fn(x_recon, x_start)
        return loss

    def loss(self, x, aux_info={}):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, t, aux_info=aux_info)

    @property
    def loss_fn(self):
        if self.loss_type == "l1":
            return F.l1_loss
        elif self.loss_type == "l2":
            return F.mse_loss
        else:
            raise ValueError(f"invalid loss type {self.loss_type}")


if __name__ == "__main__":
    diffuser = DiffuserRotationModel(
        n_timesteps=50,
        horizon=80,
        diffuser_use_preceiver=False,
    ).cuda()

    state_model = diffuser.model
    x = torch.randn(2, 40, 259).cuda()
    cond = torch.randn(2, 256).cuda()
    time = torch.randn(2).cuda()

    # Test the loss
    data_batch = {
        "gt_trajectory": torch.randn(2, 40, 3).cuda(),
        "gt_r6d_res_trajectory": torch.randn(2, 40, 6).cuda(),
        "color_aug": torch.randn(2, 3, 256, 456).cuda(),
        "object_color_aug": torch.randn(2, 3, 256, 256).cuda(),
        "depth_aug": torch.randn(2, 1, 256, 456).cuda(),
        "intrinsics": torch.randn(2, 3, 3).cuda(),
        "voxel_bounds": torch.randn(2, 2).cuda(),
        "start_pos": torch.randn(2, 3).cuda(),
        "end_pos": torch.randn(2, 3).cuda(),
        "gt_traj_min_bound": torch.randn(2, 3).cuda(),
        "gt_traj_max_bound": torch.randn(2, 3).cuda(),
        "action_feature": torch.randn(2, 512).cuda(),
        "tsdf_grid": torch.randn(2, 64, 64, 64).cuda(),
        "gt_rot_init": torch.randn(2, 3, 3).cuda(),
    }
    # for _ in range(20):
    aux_info = diffuser.get_aux_info(data_batch, training=True)
    # diffuser_guidance = DiffuserGuidance()
    # diffuser.set_guidance(diffuser_guidance)
    import time

    # with torch.no_grad():
    with torch.no_grad():
        for _ in range(100):
            time_start = time.time()
            out_info = diffuser.conditional_sample(
                data_batch=data_batch,
                aux_info=aux_info,
                apply_guidance=False,
                return_guidance_losses=False,
                guide_clean=False,
                num_samp=1,
            )
            print(out_info["pred_trajectory"].shape)
            print(out_info.keys())
            print(time.time() - time_start)
