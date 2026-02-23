import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
import diffuser_utils.dataset_utils as DatasetUtils
import diffuser_utils.tensor_utils as TensorUtils
from models.diffuser_rot import DiffuserRotationModel
from models.helpers import EMA
import open3d as o3d
import time
import cv2
import torchvision
from models.clip import clip, tokenize
import pandas as pd


GRIPPER = o3d.io.read_triangle_mesh("assets/panda_hand_mesh.obj")
VLM, VLM_TRANSFORM = clip.load("ViT-B/16", jit=False)
VLM.eval()
for p in VLM.parameters():
    p.requires_grad = False


class TrajectoryRotationDiffusionModule(pl.LightningModule):
    def __init__(self, algo_config, train_config):
        super(TrajectoryRotationDiffusionModule, self).__init__()
        self.algo_config = algo_config
        self.train_config = train_config
        self.nets = nn.ModuleDict()

        # Conditioning parsing
        self.cond_drop_obj_p = algo_config.training.conditioning_drop_object
        self.cond_drop_act_p = algo_config.training.conditioning_drop_action
        self.cond_fill_val = algo_config.training.conditioning_drop_fill

        # Initialize the diffuser
        policy_kwargs = algo_config.model
        policy_kwargs.update(dict(cond_fill_value=self.cond_fill_val))
        self.nets["policy"] = DiffuserRotationModel(**policy_kwargs)
        # set up EMA
        self.use_ema = algo_config.ema.use_ema
        if self.use_ema:
            print("DIFFUSER: using EMA... val and get_action will use ema model")
            self.ema = EMA(algo_config.ema.ema_decay)
            self.ema_policy = copy.deepcopy(self.nets["policy"])
            self.ema_policy.requires_grad_(False)
            self.ema_update_every = algo_config.ema.ema_step
            self.ema_start_step = algo_config.ema.ema_start_step
            self.reset_parameters()

        self.action_annotations = pd.read_csv(algo_config.training.action_label_path)
        self.curr_train_step = 0  # step within an epoch

    @property
    def checkpoint_monitor_keys(self):
        if self.use_ema:
            return {"valLoss": "val/ema_losses_diffusion_loss"}
        else:
            return {"valLoss": "val/losses_diffusion_loss"}

    @torch.no_grad()
    def encode_action(self, data_batch, clip_model=VLM, max_length=20):
        assert "action_tokens" in data_batch or "action_text" in data_batch
        if "action_tokens" in data_batch:
            action_tokens = data_batch["action_tokens"]  # [B, 77]
            action_feature = clip_model.encode_text(action_tokens.clone())

        else:
            action_tokens, action_feature = DatasetUtils.encode_text_clip(
                clip_model,
                [data_batch["action_text"]],
                max_length=max_length,
                device="cuda",
            )

            action_tokens.to(self.device)
            action_feature.to(self.device)
        data_batch.update({"action_feature": action_feature.float()})

    def forward(
        self,
        data_batch,
        num_samp=1,
        return_diffusion=False,
        return_guidance_losses=False,  # FIXME: this is a hack to avoid guidance
        apply_guidance=False,  # FIXME: this is a hack to avoid guidance
        class_free_guide_w=0.0,
        guide_clean=False,  # FIXME: this is a hack to avoid guidance
    ):
        curr_policy = self.nets["policy"]
        # this function is only called at validation time, so use ema
        if self.use_ema:
            curr_policy = self.ema_policy

        return curr_policy(
            data_batch,
            num_samp,
            return_diffusion=return_diffusion,
            return_guidance_losses=return_guidance_losses,
            apply_guidance=apply_guidance,
            class_free_guide_w=class_free_guide_w,
            guide_clean=guide_clean,
        )

    def reset_parameters(self):
        self.ema_policy.load_state_dict(self.nets["policy"].state_dict())

    def configure_optimizers(self):
        optim_params = self.algo_config.optimzation
        return optim.Adam(
            params=self.nets["policy"].parameters(),
            lr=optim_params.learning_rate,
        )

    def step_ema(self, step):
        if step < self.ema_start_step:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_policy, self.nets["policy"])

    def training_step_end(self, data_batch):
        self.curr_train_step += 1

    def training_step(self, data_batch, batch_idx):
        batch_size = len(data_batch["color"])
        action_tokens_null = tokenize("")
        action_tokens_null = action_tokens_null.repeat(batch_size, 1)
        action_tokens_null = action_tokens_null.to(data_batch["action_tokens"].device)

        drop_mask_obj = (
            torch.rand(len(data_batch["object_color"])) < self.cond_drop_obj_p
        )
        drop_mask_act = (
            torch.rand(len(data_batch["action_tokens"])) < self.cond_drop_act_p
        )

        data_batch["object_color"][drop_mask_obj] = self.cond_fill_val
        data_batch["object_color_aug"][drop_mask_obj] = self.cond_fill_val
        data_batch["action_tokens"][drop_mask_act] = action_tokens_null[drop_mask_act]

        # self.visualize_trajectory(data_batch)
        self.encode_action(data_batch)
        losses = self.nets["policy"].compute_losses(data_batch)

        # Summarize loss
        total_loss = 0.0
        for lk, l in losses.items():
            losses[lk] = l * self.algo_config.training.loss_weights[lk]
            total_loss += losses[lk]

        for lk, l in losses.items():
            self.log("train/losses_" + lk, l)

        # Monitor the trajectory length
        gt_traj = data_batch["gt_trajectory"]
        traj_len = torch.linalg.norm(gt_traj[:, 0] - gt_traj[:, -1], dim=-1)
        traj_len = traj_len.mean()
        self.log("train/traj_len", traj_len)
        self.training_step_end(data_batch)

        return {
            "loss": total_loss,
            "all_losses": losses,
        }

    def validation_step(self, data_batch, batch_idx):
        self.encode_action(data_batch)
        curr_policy = self.nets["policy"]
        losses = TensorUtils.detach(curr_policy.compute_losses(data_batch))
        out = curr_policy(
            data_batch,
            num_samp=1,  # self.algo_config.training.num_eval_samples,
            return_diffusion=False,
            return_guidance_losses=False,  # FIXME: this is a hack to avoid guidance
            apply_guidance=False,  # FIXME: this is a hack to avoid guidance
        )

        data_batch.update(
            {"pred_trajectories_rot": out["predictions_rot"]}
        )  # [B, N, H, 3, 3]
        return_dict = {"losses": losses}

        if self.train_config.validation.visualize_mode == "render":
            self.visualize_trajectory(
                data_batch, config_path=self.train_config.validation.render_config
            )

            pred_vis = torchvision.utils.make_grid(data_batch["pred_vis"], nrow=4)
            obj_vis = torchvision.utils.make_grid(data_batch["object_color"], nrow=4)
            self.logger.log_image("val/pred_vis", [pred_vis])
            self.logger.log_image("val/obj_vis", [obj_vis])
            return_dict.update({"pred_vis": pred_vis})
        return return_dict

    def visualize_trajectory(self, data_batch, **kwargs):
        if self.train_config.validation.visualize_mode == "render":
            self.visualize_trajectory_by_rendering(data_batch, **kwargs)
        else:
            raise ValueError(
                "Invalid visualization mode: {}".format(
                    self.train_config.validation.visualize_mode
                )
            )

    def visualize_trajectory_by_rendering(
        self,
        data_batch,
        config_path,
        window=False,
        return_vis=False,
        draw_grippers=True,
        **kwargs
    ):
        batch_size = len(data_batch["color"])
        results = []
        for i in range(batch_size):
            vis_o3d = []
            depth = data_batch["depth"][i].cpu().numpy()
            color = data_batch["color"][i].cpu().numpy().transpose(1, 2, 0)
            intr = data_batch["intrinsics"][i].cpu().numpy()
            # gt_traj = data_batch["gt_trajectory"][i].cpu().numpy()

            # backproject
            points_scene, scene_ids = DatasetUtils.backproject(
                depth,
                intr,
                depth > 0,
                NOCS_convention=False,
            )

            colors_scene = color.copy()[scene_ids[0], scene_ids[1]]
            pcd_scene = DatasetUtils.visualize_points(points_scene, colors_scene)

            vis_o3d = [pcd_scene]

            if "pred_trajectories_rot" in data_batch and draw_grippers:
                print("===> Visualizing pred trajectories")
                pred_trajs_rot = (
                    data_batch["pred_trajectories_rot"][i].cpu().numpy()[:1]
                )
                for pi, pred_traj_rot in enumerate(pred_trajs_rot):
                    traj_tra = data_batch["gt_trajectory"][i].cpu().numpy()
                    gripper_colors = DatasetUtils.get_heatmap(
                        np.arange(len(traj_tra))[None], "plasma"
                    )[0]
                    gripper_vis = []
                    for wi, wp in enumerate(traj_tra):
                        wr = pred_traj_rot[wi]
                        if wi % 10 == 0:
                            gripper_i = copy.deepcopy(GRIPPER)
                            gripper_i.scale(0.8, [0, 0, 0])
                            gripper_i.rotate(wr)
                            gripper_i.translate(wp)
                            gripper_i.paint_uniform_color(gripper_colors[wi])
                            gripper_vis.append(gripper_i)

                    gripper_vis_final = gripper_vis[0]
                    for gi in range(1, len(gripper_vis)):
                        gripper_vis_final += gripper_vis[gi]
                    vis_o3d += [gripper_vis_final]
                    del gripper_vis

            if window:
                o3d.visualization.draw(vis_o3d)

            if return_vis:
                return vis_o3d

            render_dist = np.median(np.linalg.norm(points_scene, axis=1))
            render_img = DatasetUtils.render_offscreen(
                vis_o3d,
                config_path,
                # dist=render_dist,
                resize_factor=0.5,
            )
            render_img = torchvision.transforms.ToTensor()(render_img)
            results.append(render_img)
        results = torch.stack(results, dim=0)  # [B, C, H, W]
        data_batch.update({"pred_vis": results})
