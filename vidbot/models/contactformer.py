import einops
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from vidbot.diffuser_utils.dataset_utils import compute_model_size
from vidbot.models.layers_2d import ResNet50Encoder, ResNet50Decoder, PositionalEmbeddingV2
from vidbot.models.preceiver import FeaturePreceiver


class ContactFormer(pl.LightningModule):
    def __init__(
        self,
        in_channels=3,
        out_channels=2,
        use_skip=True,
        encode_action=False,
        use_min_loss=False,
        **kwargs,
    ):
        super(ContactFormer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_skip = use_skip
        self.encode_action = encode_action
        self.use_min_loss = use_min_loss
        self.transform = transforms.Compose(
            [
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.visual = ResNet50Encoder(input_channels=in_channels)
        self.decoder = ResNet50Decoder(output_channels=out_channels, use_skip=use_skip)
        self.bottleneck_feature_key = self.decoder.bottleneck_feature_key
        self.latent_dim = self.decoder.latent_dim
        self.visual_proj = nn.Linear(self.latent_dim, 512)
        # self.proj = nn.Linear(self.latent_dim, 512)

        if self.encode_action:
            print("Encoding action")
            self.action_proj = nn.Linear(512, 512)
            self.action_fuser = FeaturePreceiver(
                transition_dim=512, condition_dim=512, time_emb_dim=0
            )
            self.final_proj = nn.Linear(self.action_fuser.last_dim, 512)
        else:
            print("No action encoding")

        self.fuser = nn.Sequential(
            nn.TransformerEncoderLayer(
                d_model=512, nhead=4, dim_feedforward=512, batch_first=True
            ),
            nn.Linear(512, self.latent_dim),
        )
        self.positional_embedding = PositionalEmbeddingV2(d_model=512, max_len=400)

    def forward(self, data_batch, training=False):
        outputs = {}
        object_color_key = "object_color"
        object_depth_key = "object_depth"
        if training:
            object_color_key += "_aug"

        inputs = self.transform(data_batch[object_color_key])
        if self.in_channels == 4:
            object_depth = data_batch[object_depth_key][:, None]
            inputs = torch.cat([inputs, object_depth], dim=1)
        features = self.visual(inputs)
        features = self.forward_latent(data_batch, features)
        pred = self.decoder(features)

        # Post-process the prediction
        pred_final = []
        pred_vfs = pred[:, : self.out_channels - 1]  # [B, 8, H, W]
        pred_mask = pred[:, self.out_channels - 1 :]  # [B, 1, H, W]
        for hi in range(0, self.out_channels - 1, 2):
            pred_vf = pred_vfs[:, hi : hi + 2]  # [B, 2, H, W]
            pred_vf = F.normalize(pred_vf, p=2, dim=1)  # [-1, 1]
            pred_vf = pred_vf.clamp(-1, 1)
            pred_final.append(pred_vf)
        pred_final.append(pred_mask)
        pred_final = torch.cat(pred_final, dim=1)
        # pred = torch.cat([pred_vf, pred[:, 2:]], dim=1)
        outputs["pred"] = pred_final  # [B, 8+1, H, W]
        return outputs

    def forward_latent(self, data_batch, features):
        latent = features[self.bottleneck_feature_key]
        h, w = latent.shape[-2:]
        latent = einops.rearrange(latent, "b c h w -> b (h w) c")
        latent = self.visual_proj(latent)
        # latent = self.proj(latent)
        if self.encode_action:
            action_feature = data_batch["action_feature"][:, None]
            action_feature = self.action_proj(action_feature)
            latent = self.action_fuser(latent, action_feature)
            latent = self.final_proj(latent)

        latent = self.positional_embedding(latent)
        latent = self.fuser(latent)
        latent = einops.rearrange(latent, "b (h w) c -> b c h w", h=h, w=w)
        features[self.bottleneck_feature_key] = latent
        return features

    def compute_losses(self, data_batch, training=True):
        losses = {}
        total_loss = 0
        outputs = self.forward(data_batch, training=training)

        pred = outputs["pred"]
        batch_size, _, h, w = pred.shape
        pred_vfs, pred_mask = (
            pred[:, : self.out_channels - 1],
            pred[:, self.out_channels - 1],
        )

        # pred_vf[:, :2] = (pred_vf[:, :2] + 1) / 2  # [-1, 1] => [0, 1]
        pred_vfs = (pred_vfs + 1) / 2  # [-1, 1] => [0, 1]

        targ_vfs = data_batch["vf_contact"]
        targ_vfs = targ_vfs.view(batch_size, -1, h, w)  # [B, 8, H, W]

        # Compute the offset loss
        num_hypo = (self.out_channels - 1) // 2
        targ_vfs = targ_vfs.view(batch_size, num_hypo, 2, h, w)  # [B, 4, 2, H, W]
        pred_vfs = pred_vfs.view(batch_size, num_hypo, 2, h, w)  # [B, 4, 2, H, W]

        if self.use_min_loss:
            offset_loss = 0
            for hi in range(num_hypo):
                pred_vf = pred_vfs[:, hi][:, None]  # [B, 1, 2, H, W]
                pred_vf = pred_vf.repeat(1, num_hypo, 1, 1, 1)  # [B, 4, 2, H, W]

                offset_loss_hi = F.l1_loss(pred_vf, targ_vfs, reduction="none")  # [B, 4, 2, H, W]
                offset_loss_hi = offset_loss_hi.view(batch_size, num_hypo, -1)  # [B, 4, 2*H*W]
                offset_loss_hi = offset_loss_hi.mean(dim=-1)  # [B, 4]
                offset_loss_hi = offset_loss_hi.min(dim=-1)[0]  # [B]
                offset_loss_hi = offset_loss_hi.mean()
                offset_loss += offset_loss_hi
            total_loss += offset_loss * 10
        else:
            offset_loss = F.l1_loss(pred_vfs, targ_vfs, reduction="none")
            offset_loss = offset_loss.mean()
            total_loss += offset_loss * 10

        # Compute the mask loss
        mask_gt = data_batch["object_mask"].float()
        mask_loss = F.binary_cross_entropy_with_logits(pred_mask, mask_gt)
        total_loss += mask_loss

        # Summarize the losses
        losses["mask_loss"] = mask_loss
        losses["offset_loss"] = offset_loss
        losses["total_loss"] = total_loss
        return losses


if __name__ == "__main__":

    model = ContactFormer(in_channels=3, out_channels=9, use_skip=True, encode_action=True)
    data_batch = {
        "object_depth": torch.rand(2, 256, 256),
        "object_mask": torch.rand(2, 256, 256),
        "object_color": torch.rand(2, 3, 256, 256),
        "object_color_aug": torch.rand(2, 3, 256, 256),
        "vf_contact": torch.rand(2, 4, 2, 256, 256),
        "action_feature": torch.rand(2, 512),
    }
    outputs = model(data_batch)
    losses = model.compute_losses(data_batch)
    compute_model_size(model)
    print(losses["total_loss"])
    print(outputs["pred"].shape)
