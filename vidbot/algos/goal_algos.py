import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision

import vidbot.diffuser_utils.dataset_utils as DatasetUtils
from vidbot.models.goalformer import GoalFormer
from vidbot.models.clip import clip, tokenize


# CLIP-based action text encoder
VLM, VLM_TRANSFORM = clip.load("ViT-B/16", jit=False)
VLM.eval()
for p in VLM.parameters():
    p.requires_grad = False


class GoalFormerModule(pl.LightningModule):
    def __init__(self, algo_config, train_config):
        super(GoalFormerModule, self).__init__()
        self.algo_config = algo_config
        self.train_config = train_config
        self.nets = nn.ModuleDict()
        policy_kwargs = algo_config.model

        self.nets["policy"] = GoalFormer(**policy_kwargs)
        self.curr_train_step = 0  # step within an epoch

        # Get the dropout rate
        self.conditioning_drop_color = algo_config.training.conditioning_drop_color
        self.conditioning_drop_bbox = algo_config.training.conditioning_drop_bbox
        self.conditioning_drop_action = algo_config.training.conditioning_drop_action
        self.drop_color_fill = algo_config.training.drop_color_fill

        # Load the action label path
        self.action_annotations = pd.read_csv("/opt/vidbot/assets/EPIC_train_action_labels.csv")

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

        if "action_tokens" in data_batch:
            # Encode the verb text
            uid = data_batch["uid"]
            verb_text = []
            for bi in range(uid.size(0)):
                uid_i = uid[bi].item()
                verb_i = self.action_annotations[self.action_annotations["uid"] == uid_i][
                    "verb"
                ].values[0]
                if verb_i not in [
                    "pick",
                    "pick-up",
                    "take",
                    "get",
                    "open",
                    "pull-out",
                    "pull",
                    "place",
                    "put-down",
                    "drop",
                    "close",
                    "push",
                ]:
                    verb_i = "other"
                else:
                    verb_i = verb_i.replace("-", "")
                verb_text.append(verb_i)
        else:
            action_text = data_batch["action_text"]
            verb_text = action_text.split(" ")[0]
            if verb_text not in [
                "pick",
                "pick-up",
                "take",
                "get",
                "open",
                "pull-out",
                "pull",
                "place",
                "put-down",
                "drop",
                "close",
                "push",
            ]:
                verb_text = "other"
            else:
                verb_text = verb_text.replace("-", "")
            verb_text = [verb_text]

        verb_tokens, verb_feature = DatasetUtils.encode_text_clip(
            clip_model,
            verb_text,
            max_length=max_length,
            device="cuda",
        )

        verb_tokens.to(self.device)
        verb_feature.to(self.device)

        data_batch.update({"action_feature": action_feature.float()})
        data_batch.update({"verb_feature": verb_feature.float()})

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
            weight_decay=optim_params.weight_decay,
        )
        return optimizer

    def training_step_end(self, data_batch):
        self.curr_train_step += 1

    def training_step(self, data_batch, batch_idx):
        batch_size = len(data_batch["color"])
        action_tokens_null = tokenize("")
        action_tokens_null = action_tokens_null.repeat(batch_size, 1)
        action_tokens_null = action_tokens_null.to(data_batch["action_tokens"].device)

        drop_mask_color = torch.rand(batch_size) < self.conditioning_drop_color
        drop_mask_action = torch.rand(batch_size) < self.conditioning_drop_action
        drop_mask_bbox = torch.rand(batch_size) < self.conditioning_drop_bbox

        # Drop the color
        if self.conditioning_drop_color > 0:
            data_batch["object_color"][drop_mask_color] = self.drop_color_fill
            data_batch["object_color_aug"][drop_mask_color] = self.drop_color_fill

        # Drop the action
        if self.conditioning_drop_action > 0:
            data_batch["action_tokens"][drop_mask_action] = action_tokens_null[drop_mask_action]

        # Drop the bbox
        if self.conditioning_drop_bbox > 0:
            data_batch["bbox"][drop_mask_bbox] = 0.0

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
        object_color = data_batch["object_color"]
        vfd_inp = data_batch["vfd"]  # [B, 3, H, W]
        goal_heatmap = data_batch["goal_heatmap"]
        start_pos_depth = data_batch["start_pos_depth"]

        pred = outputs["pred"]  # [B, 3, H, W]
        vf_gt, d_gt = vfd_inp[:, :2], vfd_inp[:, 2]
        vf_pred, d_pred, heatmap_pred = pred[:, :2], pred[:, 2], pred[:, 3].sigmoid()

        heatmap_pred_vis = self.visualize_heatmap(heatmap_pred, data_batch)
        heatmaps_gt_vis = self.visualize_heatmap(goal_heatmap, data_batch)

        # Visualize the vector field
        vf_gt = vf_gt * 2 - 1  # [0,1] -> [-1, 1]
        vf_gt_vis = self.visualize_vector_field(vf_gt)
        vf_pred_vis = self.visualize_vector_field(vf_pred)  # already between [-1, 1]
        goal_coord_gt_vis = self.visualize_fitted_goal_coords(
            vf_gt, data_batch, outputs, sigma_scale=1.0
        )
        goal_coord_pred_vis = self.visualize_fitted_goal_coords(
            vf_pred, data_batch, outputs, sigma_scale=0.1
        )
        start_pos_depth_gt_vis = self.visualize_goal_depth(start_pos_depth)
        end_pos_depth_gt_vis = self.visualize_goal_depth(d_gt)
        end_pos_depth_pred_vis = self.visualize_goal_depth(d_pred)

        vf_gt_vis = torchvision.utils.make_grid(vf_gt_vis, nrow=4)
        vf_pred_vis = torchvision.utils.make_grid(vf_pred_vis, nrow=4)
        goal_coord_gt_vis = torchvision.utils.make_grid(goal_coord_gt_vis, nrow=4)
        goal_coord_pred_vis = torchvision.utils.make_grid(goal_coord_pred_vis, nrow=4)

        start_pos_depth_gt_vis = torchvision.utils.make_grid(start_pos_depth_gt_vis, nrow=4)
        end_pos_depth_gt_vis = torchvision.utils.make_grid(end_pos_depth_gt_vis, nrow=4)
        end_pos_depth_pred_vis = torchvision.utils.make_grid(end_pos_depth_pred_vis, nrow=4)

        object_color_vis = torchvision.utils.make_grid(object_color, nrow=4)
        heatmap_pred_vis = torchvision.utils.make_grid(heatmap_pred_vis, nrow=4)
        heatmaps_gt_vis = torchvision.utils.make_grid(heatmaps_gt_vis, nrow=4)

        self.logger.log_image("val_vis/d_gt", [end_pos_depth_gt_vis], step=self.curr_train_step)
        self.logger.log_image(
            "val_vis/d_pred", [end_pos_depth_pred_vis], step=self.curr_train_step
        )
        self.logger.log_image(
            "val_vis/start_d_gt", [start_pos_depth_gt_vis], step=self.curr_train_step
        )

        self.logger.log_image("val_vis/vf_gt", [vf_gt_vis], step=self.curr_train_step)
        self.logger.log_image("val_vis/vf_pred", [vf_pred_vis], step=self.curr_train_step)
        self.logger.log_image(
            "val_vis/goal_coords_pred", [goal_coord_pred_vis], step=self.curr_train_step
        )

        self.logger.log_image(
            "val_vis/goal_coords_gt", [goal_coord_gt_vis], step=self.curr_train_step
        )

        self.logger.log_image(
            "val_vis/object_color", [object_color_vis], step=self.curr_train_step
        )

        self.logger.log_image(
            "val_vis/heatmap_pred", [heatmap_pred_vis], step=self.curr_train_step
        )
        self.logger.log_image("val_vis/heatmap_gt", [heatmaps_gt_vis], step=self.curr_train_step)

        # Compute the depth error
        depth_error = F.l1_loss(d_pred, d_gt)
        self.log("val/depth_error", depth_error)
        self.validate_goal_depth_direction_accuracy(data_batch, outputs)

    def validate_goal_depth_direction_accuracy(self, data_batch, outputs):
        uid = data_batch["uid"]
        correct_sign = 0
        for bi in range(uid.size(0)):
            uid_i = uid[bi].item()
            verb_i = self.action_annotations[self.action_annotations["uid"] == uid_i][
                "verb"
            ].values[0]
            pred_end_pos_depth_i = outputs["pred"][bi, 2]
            start_pos_depth_i = data_batch["start_pos_depth"][bi]
            end_pos_depth_i = data_batch["end_pos_depth"][bi]
            depth_diff = (pred_end_pos_depth_i - start_pos_depth_i).mean()
            depth_diff_gt = (end_pos_depth_i - start_pos_depth_i).mean()
            if verb_i in [
                "pick",
                "pick-up",
                "take",
                "get",
                "open",
                "pull-out",
                "pull",
            ]:
                if depth_diff_gt < 0:
                    correct_sign += 1
            elif verb_i in [
                "place",
                "put-down",
                "drop",
                "close",
                "push",
            ]:
                if depth_diff_gt > 0:
                    correct_sign += 1
            else:
                if torch.sign(depth_diff) == torch.sign(depth_diff_gt):
                    correct_sign += 1
        acc = correct_sign / uid.size(0)
        self.log("val/depth_direction_acc", acc)

    def visualize_goal_depth(self, depth, height=256, width=448, depth_min=0.2, depth_max=2.0):
        depth = depth.view(-1, height, width)
        depth = depth.clamp(min=depth_min, max=depth_max)
        results = []
        for bi in range(depth.size(0)):
            depth_i = depth[bi].cpu().numpy()
            depth_i = (depth_i - depth_min) / (depth_max - depth_min)
            colormaps = plt.cm.get_cmap("turbo")
            depth_i = colormaps(depth_i)[..., :3]  # don't need alpha channel
            depth_i = torchvision.transforms.ToTensor()(depth_i)
            results.append(depth_i)
        results = torch.stack(results, dim=0)
        return results

    def visualize_uncertainty(self, uncertainty):
        results = []
        for bi in range(uncertainty.size(0)):
            unc_i = uncertainty[bi].cpu().numpy()[0]  # [H, W]
            unc_i = DatasetUtils.get_heatmap(unc_i)
            unc_i = torchvision.transforms.ToTensor()(unc_i)
            results.append(unc_i)
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

    def visualize_features(self, features, colors):
        results = []
        features = torch.nn.functional.interpolate(
            features, size=(colors.shape[-2], colors.shape[-1]), mode="bilinear"
        )

        for bi in range(features.size(0)):
            feat_i = features[bi].cpu().permute(1, 2, 0)  # []
            vis_i = DatasetUtils.apply_pca_colormap(feat_i)
            vis_i = vis_i.permute(2, 0, 1)
            results.append(vis_i)
        results = torch.stack(results, dim=0)
        return results

    def visualize_fitted_goal_coords(self, vf, data_batch, outputs, sigma_scale=0.75):
        color = data_batch["color"]
        uid = data_batch["uid"]
        results = []
        for bi in range(vf.size(0)):
            vf_i = vf[bi].cpu().numpy().transpose(1, 2, 0)
            color_i = color[bi].cpu().numpy().transpose(1, 2, 0)
            color_i = (color_i * 255).astype(np.uint8)
            vis_i = color_i[..., [2, 1, 0]].copy()
            uid_i = uid[bi].item()
            narration_i = self.action_annotations[self.action_annotations["uid"] == uid_i][
                "narration"
            ].values[0]
            verb_i = self.action_annotations[self.action_annotations["uid"] == uid_i][
                "verb"
            ].values[0]
            pred_end_pos_depth_i = outputs["pred"][bi, 2]
            start_pos_depth_i = data_batch["start_pos_depth"][bi]
            end_pos_depth_i = data_batch["end_pos_depth"][bi]
            depth_diff = (pred_end_pos_depth_i - start_pos_depth_i).mean()
            depth_diff_gt = (end_pos_depth_i - start_pos_depth_i).mean()
            is_good_depth = False
            if verb_i in [
                "pick",
                "pick-up",
                "take",
                "get",
                "open",
                "pull-out",
                "pull",
            ]:
                if depth_diff_gt < 0:
                    is_good_depth = True
            elif verb_i in [
                "place",
                "put-down",
                "drop",
                "close",
                "push",
            ]:
                if depth_diff_gt > 0:
                    is_good_depth = True
            else:
                if torch.sign(depth_diff) == torch.sign(depth_diff_gt):
                    is_good_depth = True
            if is_good_depth:
                narration_i = "{}: GOOD DEPTH!".format(narration_i)
            else:
                narration_i = "{}: BAD DEPTH!".format(narration_i)
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
            uv, sigma, win_hypothesis, win_hypotheses, inlier_ratio = (
                DatasetUtils.ransac_voting_layer(
                    grid_torch,
                    vfield_torch,
                    inlier_threshold=0.99,
                )
            )

            # Sample from the fitted distribution and draw on the image
            sampled_uvs = np.random.multivariate_normal(uv, sigma * sigma_scale, 1000)
            for sampled_uv in sampled_uvs:
                cv2.circle(
                    vis_i,
                    (int(sampled_uv[0]), int(sampled_uv[1])),
                    3,
                    (255, 0, 0),
                    -1,
                )
            cv2.circle(vis_i, (int(uv[0]), int(uv[1])), 6, (0, 255, 0), -1)
            if is_good_depth:
                text_color = (0, 255, 0)
            else:
                text_color = (0, 0, 255)
            cv2.putText(
                vis_i,
                narration_i,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                text_color,
                2,
            )
            vis_i = vis_i[..., [2, 1, 0]]
            vis_i = torchvision.transforms.ToTensor()(vis_i)
            results.append(vis_i)
        results = torch.stack(results, dim=0)
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

    def visualize_heatmap(self, score, data_batch):
        color = data_batch["color"]
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
