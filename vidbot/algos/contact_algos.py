import numpy as np
import copy

from pytorch_lightning.core.optimizer import LightningOptimizer
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F

import diffuser_utils.dataset_utils as DatasetUtils
import torchvision
import cv2
from algos.vq_algos import GoalVectorQuantizationModule
from options import load_options
from models.contactformer import ContactFormer
import pandas as pd
from models.clip.clip import tokenize
from copy import deepcopy
from models.gpt import GPT, GPTConfig

from models.clip import clip, tokenize

# CLIP-based action text encoder
VLM, VLM_TRANSFORM = clip.load("ViT-B/16", jit=False)
VLM.eval()
for p in VLM.parameters():
    p.requires_grad = False


class ContactFormerModule(pl.LightningModule):
    def __init__(self, algo_config, train_config):
        super(ContactFormerModule, self).__init__()
        self.algo_config = algo_config
        self.train_config = train_config
        self.nets = nn.ModuleDict()

        # Initialize the contact former
        policy_kwargs = algo_config.model

        self.nets["policy"] = ContactFormer(**policy_kwargs)
        self.conditioning_drop_action = algo_config.training.conditioning_drop_action

        # Load the action label path
        self.curr_train_step = 0

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

    def forward(self, data_batch, training=False):
        # self.encode_action(data_batch)
        curr_policy = self.nets["policy"]
        outputs = curr_policy(data_batch, training)
        return outputs

    def configure_optimizers(self):
        policy = self.nets["policy"]
        optim_params = self.algo_config.optimzation
        optimizer = optim.Adam(
            params=policy.parameters(),
            lr=optim_params.learning_rate,
            betas=optim_params.betas,
            weight_decay=optim_params.weight_decay,
        )
        return optimizer

    def training_step_end(self, data_batch):
        self.curr_train_step += 1

    def training_step(self, data_batch, batch_idx):

        batch_size = len(data_batch["object_color"])
        action_tokens_null = tokenize("")
        action_tokens_null = action_tokens_null.repeat(batch_size, 1)
        action_tokens_null = action_tokens_null.to(data_batch["action_tokens"].device)
        drop_mask_action = torch.rand(batch_size) < self.conditioning_drop_action

        # Drop the action
        if self.conditioning_drop_action > 0:
            data_batch["action_tokens"][drop_mask_action] = action_tokens_null[
                drop_mask_action
            ]

        self.encode_action(data_batch)
        losses = self.nets["policy"].compute_losses(data_batch, training=True)
        total_loss = losses["total_loss"]
        for lk, l in losses.items():
            self.log("train/{}".format(lk), l)

        self.training_step_end(data_batch)
        return {"loss": total_loss, "all_losses": losses}

    def validation_step(self, data_batch, batch_idx):
        print("====> Validation step")
        self.encode_action(data_batch)
        curr_policy = self.nets["policy"]
        outputs = curr_policy(data_batch, training=False)
        losses = curr_policy.compute_losses(data_batch, training=False)
        # self.visualize_goal_image(data_batch)
        for lk, l in losses.items():
            self.log("val/{}".format(lk), l)

        # Visualize the results
        num_hypo = (self.nets["policy"].out_channels - 1) // 2
        vf_genes = outputs["pred"][:, : num_hypo * 2]  # [B, 8, H, W]
        vf_genes = vf_genes.view(-1, num_hypo, 2, vf_genes.size(2), vf_genes.size(3))

        object_mask_pred = (outputs["pred"][:, -1] > 0).float()  # [B, 1, H, W]
        object_scores_pred = outputs["pred"][:, -1].sigmoid()
        object_scores_gt = data_batch["object_mask"].float()[:, None]

        for hi in range(4):
            vf_inp = data_batch["vf_contact"][:, hi]  # [B, 2, H, W]
            vf_inp = vf_inp * 2 - 1  # [0, 1] -> [-1, 1]
            vf_gene = vf_genes[:, hi]  # [B, 2, H, W]

            vf_vis = self.visualize_vector_field(vf_inp)
            vf_gene_vis = self.visualize_vector_field(
                vf_gene
            )  # already between [-1, 1]

            vf_vis = torchvision.utils.make_grid(vf_vis, nrow=4)
            vf_gene_vis = torchvision.utils.make_grid(vf_gene_vis, nrow=4)
            contact_coord_gt_vis = self.visualize_fitted_goal_coords(
                vf_inp, data_batch, sigma_scale=1.0, mask=data_batch["object_mask"]
            )

            contact_coord_pred_vis = self.visualize_fitted_goal_coords(
                vf_gene, data_batch, sigma_scale=0.1, mask=data_batch["object_mask"]
            )

            contact_coord_gt_vis = torchvision.utils.make_grid(
                contact_coord_gt_vis, nrow=4
            )
            contact_coord_pred_vis = torchvision.utils.make_grid(
                contact_coord_pred_vis, nrow=4
            )

            if hi == 0:
                self.logger.log_image(
                    "val_vis/vf_{}".format(hi), [vf_vis], step=self.curr_train_step
                )
                self.logger.log_image(
                    "val_vis/contact_coords_gt_{}".format(hi),
                    [contact_coord_gt_vis],
                    step=self.curr_train_step,
                )

            self.logger.log_image(
                "val_vis/vf_pred_{}".format(hi),
                [vf_gene_vis],
                step=self.curr_train_step,
            )
            self.logger.log_image(
                "val_vis/contact_coords_pred_{}".format(hi),
                [contact_coord_pred_vis],
                step=self.curr_train_step,
            )

        # Visualize the vector field
        # goal_vis = data_batch["goal_vis"]
        scores_pred_vis = self.visualize_heatmap(object_scores_pred, data_batch)
        scores_gt_vis = self.visualize_heatmap(object_scores_gt, data_batch)

        # goal_vis = torchvision.utils.make_grid(goal_vis, nrow=4)
        scores_pred_vis = torchvision.utils.make_grid(scores_pred_vis, nrow=4)
        scores_gt_vis = torchvision.utils.make_grid(scores_gt_vis, nrow=4)

        object_color = data_batch["object_color"]
        object_color_vis = torchvision.utils.make_grid(object_color, nrow=4)

        object_mask = data_batch["object_mask"][:, None]
        object_mask_vis = torchvision.utils.make_grid(object_mask, nrow=4)

        object_mask_pred_vis = torchvision.utils.make_grid(
            object_mask_pred[:, None], nrow=4
        )

        self.logger.log_image(
            "val_vis/object_color", [object_color_vis], step=self.curr_train_step
        )
        self.logger.log_image(
            "val_vis/object_mask", [object_mask_vis], step=self.curr_train_step
        )
        self.logger.log_image(
            "val_vis/object_mask_pred",
            [object_mask_pred_vis],
            step=self.curr_train_step,
        )
        self.logger.log_image(
            "val_vis/scores_pred", [scores_pred_vis], step=self.curr_train_step
        )
        self.logger.log_image(
            "val_vis/scores_gt", [scores_gt_vis], step=self.curr_train_step
        )

    def visualize_heatmap(self, score, data_batch):
        color = data_batch["object_color"]
        results = []
        for bi in range(score.size(0)):
            score_i = score[bi].cpu().numpy().squeeze()
            heatmap_i = DatasetUtils.get_heatmap(score_i)
            vis_i = color[bi].cpu().numpy().transpose(1, 2, 0)
            vis_i = (vis_i * 255).astype(np.uint8)
            heatmap_i = (heatmap_i * 255).astype(np.uint8)
            vis_i = cv2.addWeighted(vis_i, 0.4, heatmap_i, 0.6, 0)
            vis_i = torchvision.transforms.ToTensor()(vis_i)
            results.append(vis_i)
        results = torch.stack(results, dim=0)
        return results

    def visualize_vector_field(self, vf):
        results = []
        for bi in range(vf.size(0)):
            vf_i = vf[bi].cpu().numpy().transpose(1, 2, 0)
            vis_i = DatasetUtils.visualize_vector_field(vf_i)  # [H, W, 3]
            vis_i = torchvision.transforms.ToTensor()(vis_i)  # [3, H, W]
            results.append(vis_i)
        results = torch.stack(results, dim=0)  # [B, 3, H, W]
        return results

    def visualize_fitted_goal_coords(self, vf, data_batch, sigma_scale=0.75, mask=None):
        color = data_batch["object_color"]
        h, w = vf.size(2), vf.size(3)
        # uid = data_batch["uid"]
        results = []
        for bi in range(vf.size(0)):
            vf_i = vf[bi].cpu().numpy().transpose(1, 2, 0)
            color_i = color[bi].cpu().numpy().transpose(1, 2, 0)
            color_i = (color_i * 255).astype(np.uint8)
            vis_i = color_i[..., [2, 1, 0]].copy()

            # Do voting ...
            height, width = vf_i.shape[:2]
            grid = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
            grid = np.stack(grid, axis=-1) + 0.5  # [H, W, 2], last dim is (x, y)
            grid_flatten = grid.reshape(-1, 2)  # [H*W, 2]
            vfield_flatten = vf_i[..., :2].reshape(-1, 2)  # [H*W, 2]

            # RANSAC voting to find the goal pixel
            grid_torch = torch.from_numpy(grid_flatten).float()  # [H, W, 2]
            vfield_torch = torch.from_numpy(vfield_flatten).float()  # [H, W, 2]
            vfield_torch = F.normalize(vfield_torch, dim=-1, p=2)  # [H, W, 2]
            if mask is not None:
                mask_torch = mask[bi].squeeze()
            else:
                mask_torch = None
            uv, sigma, win_hypothesis, win_hypotheses, inlier_ratio = (
                DatasetUtils.ransac_voting_layer(
                    grid_torch,
                    vfield_torch,
                    mask_torch,
                    inlier_threshold=0.99,
                )
            )

            # Sample from the fitted distribution and draw on the image
            sampled_uvs = np.random.multivariate_normal(uv, sigma * sigma_scale, 1000)
            for sampled_uv in sampled_uvs:
                sampled_u = int(sampled_uv[0])
                sampled_v = int(sampled_uv[1])
                sampled_u = np.clip(sampled_u, 0, w - 1)
                sampled_v = np.clip(sampled_v, 0, h - 1)
                cv2.circle(
                    vis_i,
                    (sampled_u, sampled_v),
                    3,
                    (255, 0, 0),
                    -1,
                )
            cv2.circle(vis_i, (int(uv[0]), int(uv[1])), 6, (0, 255, 0), -1)

            vis_i = vis_i[..., [2, 1, 0]]
            vis_i = torchvision.transforms.ToTensor()(vis_i)
            results.append(vis_i)
        results = torch.stack(results, dim=0)
        return results

    # def visualize_goal_image(self, data_batch):
    #     # Visualize the goal
    #     color = data_batch["object_color"]
    #     vf_gt = data_batch["vf_contact"][:, :2]
    #     vf_gt = vf_gt * 2 - 1
    #     vf_gt_vis = self.visualize_vector_field(vf_gt)  # [B, 3, H, W]
    #     results = []
    #     for bi in range(color.size(0)):
    #         color_i = color[bi].cpu().numpy().transpose(1, 2, 0)
    #         vf_gt_i = vf_gt_vis[bi].cpu().numpy().transpose(1, 2, 0)
    #         color_i = (color_i * 255).astype(np.uint8)
    #         vf_gt_i = (vf_gt_i * 255).astype(np.uint8)
    #         vis_i = cv2.addWeighted(color_i, 0.4, vf_gt_i, 0.6, 0)
    #         vis_i = torchvision.transforms.ToTensor()(vis_i)
    #         results.append(vis_i)
    #     results = torch.stack(results, dim=0)  # [B, 3, H, W]
    #     data_batch.update({"goal_vis": results})
