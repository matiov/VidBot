import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
from models.rqvae import RQVAE
import diffuser_utils.dataset_utils as DatasetUtils
import torchvision
import cv2


class GoalVectorQuantizationModule(pl.LightningModule):
    def __init__(self, algo_config, train_config):
        super(GoalVectorQuantizationModule, self).__init__()
        self.algo_config = algo_config
        self.train_config = train_config
        self.nets = nn.ModuleDict()

        # Initialize the diffuser
        policy_kwargs = algo_config.model
        self.nets["policy"] = RQVAE(**policy_kwargs)

        self.curr_train_step = 0  # step within an epoch

    def forward(self, data_batch, key="vfd", return_rec=False):
        curr_policy = self.nets["policy"]
        outputs = curr_policy(data_batch, key=key, return_rec=return_rec)
        return outputs

    def configure_optimizers(self):
        optim_params = self.algo_config.optimzation
        return optim.Adam(
            params=self.nets["policy"].parameters(),
            lr=optim_params.learning_rate,
            betas=optim_params.betas,
            weight_decay=optim_params.weight_decay,
        )

    def training_step_end(self, data_batch):
        self.curr_train_step += 1

    def training_step(self, data_batch, batch_idx):
        # self.visualize_trajectory(data_batch)
        losses = self.nets["policy"].compute_losses(data_batch, key="vfd")
        total_loss = losses["total_loss"]
        for lk, l in losses.items():
            self.log("train/{}".format(lk), l)
        losses.update({"loss": total_loss, "all_losses": losses})
        self.training_step_end(data_batch)
        return losses

    def validation_step(self, data_batch, batch_idx):
        print("====> Validation step")
        curr_policy = self.nets["policy"]
        outputs = curr_policy(data_batch, key="vfd", return_rec=True)
        vfd_inp = data_batch["vfd"]  # [B, 3, H, W]
        vfd_rec = outputs["vfd_rec"]  # [B, 3, H, W]
        vf_inp, d_inp = vfd_inp[:, :2], vfd_inp[:, 2]
        vf_rec, d_rec = vfd_rec[:, :2], vfd_rec[:, 2]
        self.visualize_goal_image(data_batch)

        # Visualize the vector field
        vf_inp = vf_inp * 2 - 1  # [0,1] -> [-1, 1]
        vfd_vis = self.visualize_vector_field(vf_inp)
        vdf_rec_vis = self.visualize_vector_field(vf_rec)  # already between [-1, 1]
        goal_vis = data_batch["goal_vis"]
        vfd_vis = torchvision.utils.make_grid(vfd_vis, nrow=4)
        vdf_rec_vis = torchvision.utils.make_grid(vdf_rec_vis, nrow=4)
        goal_vis = torchvision.utils.make_grid(goal_vis, nrow=4)
        self.logger.log_image("val_vis/vfd", [vfd_vis], step=self.curr_train_step)
        self.logger.log_image(
            "val_vis/vfd_rec", [vdf_rec_vis], step=self.curr_train_step
        )
        self.logger.log_image(
            "val_vis/color_vfd", [goal_vis], step=self.curr_train_step
        )

        # Compute the depth error
        inv_depth = 1 / data_batch["depth"]  # [B, H, W]
        batch_size = inv_depth.shape[0]
        inv_depth = inv_depth.view(batch_size, -1)  # [B, H * W]
        inv_depth_min = inv_depth.min(dim=1)[0]  # [B]
        inv_depth_max = inv_depth.max(dim=1)[0]  # [B]

        d_inp = d_inp.view(batch_size, -1).mean(dim=1)  # [B]
        d_rec = d_rec.view(batch_size, -1).mean(dim=1)  # [B]
        inv_depth_inp = inv_depth_min + (inv_depth_max - inv_depth_min) * d_inp
        inv_depth_rec = inv_depth_min + (inv_depth_max - inv_depth_min) * d_rec
        depth_error = (1 / inv_depth_inp - 1 / inv_depth_rec).abs().mean()
        self.log("val/depth_error", depth_error)

    def visualize_vector_field(self, vf):
        results = []
        for bi in range(vf.size(0)):
            vf_i = vf[bi].cpu().numpy().transpose(1, 2, 0)
            vis_i = DatasetUtils.visualize_vector_field(vf_i)  # [H, W, 3]
            vis_i = torchvision.transforms.ToTensor()(vis_i)  # [3, H, W]
            results.append(vis_i)
        results = torch.stack(results, dim=0)  # [B, 3, H, W]
        return results

    def visualize_goal_image(self, data_batch):
        # Visualize the goal
        color = data_batch["color"]
        vf_gt = data_batch["vfd"][:, :2]
        vf_gt = vf_gt * 2 - 1
        vf_gt_vis = self.visualize_vector_field(vf_gt)  # [B, 3, H, W]
        results = []
        for bi in range(color.size(0)):
            color_i = color[bi].cpu().numpy().transpose(1, 2, 0)
            vf_gt_i = vf_gt_vis[bi].cpu().numpy().transpose(1, 2, 0)
            color_i = (color_i * 255).astype(np.uint8)
            vf_gt_i = (vf_gt_i * 255).astype(np.uint8)
            vis_i = cv2.addWeighted(color_i, 0.4, vf_gt_i, 0.6, 0)
            vis_i = torchvision.transforms.ToTensor()(vis_i)
            results.append(vis_i)
        results = torch.stack(results, dim=0)  # [B, 3, H, W]
        data_batch.update({"goal_vis": results})
