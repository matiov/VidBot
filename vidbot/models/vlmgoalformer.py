import einops
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from vidbot.models.gpt import GPT, GPTConfig
from vidbot.models.layers_2d import MLP, Decoder
from vidbot.models.clip import clip


class VLMGoalFormer(pl.LightningModule):
    def __init__(
        self,
        gpt=None,
        in_channels=3,
        out_channels=3,
        resolution=[256, 448],
        bbox_feature_dim=64,
        clip_model="ViT-B/16",
        freeze_visual=True,
        encode_action=False,
        encode_bbox=False,
        encode_object=False,
        num_heads_attention=4,
        num_layers_attention=2,
        object_encode_mode="roi_pool",
        **kwargs,
    ):
        super().__init__()

        self.gpt = gpt
        self.freeze_visual = freeze_visual
        self.clip_model = clip_model

        self.encode_action = encode_action
        self.encode_bbox = encode_bbox
        self.encode_object = encode_object

        self.bbox_feature_dim = bbox_feature_dim
        self.object_encode_mode = object_encode_mode

        self.visual, self.transform = clip.load(clip_model, device="cuda")
        # self.vlm, self.transform = clip.load(clip_model, device="cuda")
        # self.vlm_dim = self.vlm.visual.output_dim
        self.visual.float()

        # self.visual = nn.Sequential(*list(resnet.children())[:-2])
        # self.visual = deepcopy(self.vlm)
        self.visual_feature_dim = self.visual.visual.output_dim

        # for param in self.vlm.parameters():
        #     param.requires_grad = False

        if self.freeze_visual:
            print("... Freezing the visual backbone")
            for param in self.visual.parameters():
                param.requires_grad = False

        if self.clip_model in ["ViT-B/16"]:
            self.visual_conv = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    self.visual_feature_dim,
                    self.visual_feature_dim,
                    kernel_size=1,
                    stride=2,
                ),
            )
        else:
            self.visual_conv = nn.Identity()

        if self.encode_object:
            if self.object_encode_mode == "vlm":
                obj_dim = self.vlm_dim
            else:
                obj_dim = self.visual_feature_dim

            self.object_encode_module = MLP(
                input_dim=obj_dim,
                output_dim=self.visual_feature_dim,
                layer_dims=[self.visual_feature_dim] * 2,
            )
            self.object_attn_module = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=self.visual_feature_dim,  # 512
                    nhead=num_heads_attention,
                    dropout=0.1,
                    batch_first=True,
                ),
                num_layers=num_layers_attention,
            )

        if self.encode_action:
            self.action_encode_module = MLP(
                input_dim=self.visual_feature_dim,
                output_dim=self.visual_feature_dim,
                layer_dims=[self.visual_feature_dim] * 2,
            )
            self.action_attn_mdoule = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=self.visual_feature_dim,  # 512
                    nhead=num_heads_attention,
                    dropout=0.1,
                    batch_first=True,
                ),
                num_layers=num_layers_attention,
            )

        if self.encode_bbox:
            self.bbox_encode_module = MLP(
                input_dim=4, layer_dims=[bbox_feature_dim], output_dim=bbox_feature_dim
            )

        visual_mlp_dim_in = self.visual_feature_dim
        if self.encode_action:
            visual_mlp_dim_in += self.visual_feature_dim
        if self.encode_object:
            visual_mlp_dim_in += self.visual_feature_dim
        if self.encode_bbox:
            visual_mlp_dim_in += self.bbox_feature_dim

        self.visual_mlp = MLP(
            input_dim=visual_mlp_dim_in,
            output_dim=self.visual_feature_dim,
            layer_dims=[self.visual_feature_dim] * 2,
        )

        # 1/16, 1/8, 1/4, 1/2, 1 <=> 512, 256, 128, 64, 64
        self.predictor = Decoder(
            ch=64,
            ch_mult=[1, 2, 2, 4, 8],
            num_res_blocks=2,
            attn_resolutions=(16,),
            in_channels=in_channels,
            out_ch=out_channels,
            resolution=resolution,
            double_z=False,
            z_channels=self.visual_feature_dim,
        )

    def forward(self, data_batch, training=False):
        target_key = "vfd"
        color_key = "color"
        object_color_key = "object_color"
        if training:
            color_key += "_aug"
            object_color_key += "_aug"

        inputs = self.transform(data_batch[color_key])
        batch_size = inputs.shape[0]
        h_in, w_in = inputs.shape[-2:]
        h_out, w_out = h_in // 16, w_in // 16
        bbox = data_batch["bbox"]  # [B, 4]
        bbox_batch_id = torch.arange(batch_size, device=inputs.device)  # Only one box per sample
        bbox = torch.cat([bbox_batch_id[:, None], bbox], dim=1)  # [B, 5]
        context_feature = self.visual.get_patch_encodings(inputs)

        if self.clip_model.startswith("ViT"):
            _h_out = h_in // self.visual.visual.patch_size
            _w_out = w_in // self.visual.visual.patch_size
        elif self.clip_model.startswith("RN"):
            # _h_out = max(h_in / w_in, 1.0) * (self.visual.visual.attnpool.spacial_dim+1)
            # _w_out = max(w_in / h_in, 1.0) * (self.visual.visual.attnpool.spacial_dim+1)
            # _h_out, _w_out = int(_h_out), int(_w_out)
            _h_out, _w_out = h_out, w_out
        else:
            raise ValueError(f"Unknown CLIP model name: {self.clip_model}")

        feature = context_feature.float()
        feature_vlm = einops.rearrange(feature.clone(), "b (h w) c -> b c h w", h=_h_out, w=_w_out)

        # Acquire bbox features
        if self.encode_object:
            if self.object_encode_mode == "vlm":
                # object_color = data_batch[object_color_key]  # [B, 3, H, W]
                # object_feature = self.vlm.encode_image(self.transform(object_color))[
                #     :, None
                # ]  # [B, 1, c]
                print("... Try another way to encode object")
            elif self.object_encode_mode in ["roi_pool", "roi_align"]:
                roi_res = 6

                roi_method = eval(self.object_encode_mode)
                context_feature_obj = einops.rearrange(
                    context_feature, "b (h w) c -> b c h w", h=h_out, w=w_out
                )
                spatial_scale = context_feature_obj.shape[-1] / inputs.shape[-1]
                assert spatial_scale == context_feature_obj.shape[-2] / inputs.shape[-2]

                context_feature_obj = roi_method(
                    context_feature_obj,
                    bbox,
                    spatial_scale=spatial_scale,
                    output_size=(roi_res, roi_res),
                )  # [B, c, roi_res, roi_res]
                object_feature = einops.rearrange(context_feature_obj, "b c h w -> b (h w) c")
            else:
                raise NotImplementedError("Object encode mode not implemented")

            object_feature = self.object_encode_module(object_feature)
            context_feature_obj = self.object_attn_module(context_feature, object_feature)
            feature = torch.cat([feature, context_feature_obj], dim=-1)

        if self.encode_action:
            # assert "action_text" in data_batch or "action_tokens" in data_batch

            # if "action_tokens" in data_batch:
            #     action_tokens = data_batch["action_tokens"]  # [B, 77]
            #     action_feature = self.vlm.encode_text(action_tokens)[:, None]

            # else:
            #     action_text = data_batch["action_text"]
            #     action_tokens = tokenize(action_text).to(self.device)
            #     action_feature = self.vlm.encode_text(action_tokens)[:, None]
            action_feature = data_batch["action_feature"][:, None]
            action_feature = self.action_encode_module(action_feature)  # [B, hw, c]
            context_feature_act = self.action_attn_mdoule(context_feature, action_feature)
            feature = torch.cat([feature, context_feature_act], dim=-1)

        if self.encode_bbox:
            bbox_norm = bbox[:, 1:].clone()
            bbox_norm[:, [0, 2]] = bbox_norm[:, [0, 2]] / inputs.shape[-1]
            bbox_norm[:, [1, 3]] = bbox_norm[:, [1, 3]] / inputs.shape[-2]
            bbox_feature = self.bbox_encode_module(bbox_norm)  # [B, 64]
            bbox_feature = bbox_feature.unsqueeze(1).repeat(
                1, context_feature.shape[1], 1
            )  # [B, hw, 64]
            feature = torch.cat([feature, bbox_feature], dim=-1)  # [B, hw, c+64]

        feature = self.visual_mlp(feature)  # [B, hw, c]

        if self.gpt is not None:
            feature = self.gpt(feature)

        feature = einops.rearrange(feature, "b (h w) c -> b c h w", h=h_out, w=w_out)
        pred = self.predictor(feature)

        # Postprocess the output
        if pred is not None:
            pred_vf, pred_d = pred[:, :2], pred[:, 2:]
            pred_vf = F.normalize(pred_vf, p=2, dim=1)  # [-1, 1]
            pred_d = pred_d.tanh()  # [-1, 1]
            pred_d = (pred_d + 1) / 2  # [-1, 1] => [0, 1]
            pred = torch.cat([pred_vf, pred_d], dim=1)  # [B, 3, h, w]

        if target_key == "vfd":
            assert (
                pred[:, :2].min() >= -1 and pred[:, :2].max() <= 1
            ), "VF output should be in the range [-1, 1]"
            assert (
                pred[:, 2].min() >= 0 and pred[:, 2].max() <= 1
            ), "Invdepth factor output should be in the range [0, 1]"

        outputs = {"pred": pred, "feature": feature, "feature_vlm": feature_vlm}

        return outputs

    def compute_losses(self, data_batch, training=True, return_rec=False):
        losses = {}
        total_loss = 0
        target_key = "vfd"
        outputs = self.forward(data_batch, training=training)

        pred = outputs["pred"]
        if target_key == "vfd":
            pred[:, :2] = (pred[:, :2] + 1) / 2  # [-1, 1] => [0, 1]

        targ = data_batch[target_key]
        assert pred.min() >= 0.0 and pred.max() <= 1.0, "Output should be in [0, 1]"
        assert targ.min() >= 0.0 and targ.max() <= 1.0, "Target should be in [0, 1]"

        # Compute the offset loss
        # offset_loss = F.l1_loss(pred, targ)
        offset_loss = F.smooth_l1_loss(pred, targ, beta=1, reduction="none")
        offset_loss = offset_loss.mean()

        offset_loss = offset_loss
        total_loss += offset_loss

        losses["offset_loss"] = offset_loss
        losses["total_loss"] = total_loss
        return losses


if __name__ == "__main__":

    from easydict import EasyDict as edict

    # # Define the configuration for the GPT
    gpt_config = {
        "block_size": 768,
        "input_dim": 512,
        "output_dim": 512,
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
        "dropout": 0.1,
    }

    gpt_config = edict(gpt_config)
    gpt = GPT(GPTConfig(**gpt_config))
    goalformer = VLMGoalFormer(gpt).cuda()
    image = torch.randn(1, 3, 256, 448).cuda()
    vfd = torch.rand(1, 3, 256, 448).cuda()
    data_batch = {
        "color": image,
        "vfd": vfd,
        "bbox": torch.zeros(1, 4).cuda(),
        "object_color": torch.rand(1, 3, 256, 256).cuda(),
        "action_tokens": torch.rand(1, 77).long().cuda(),
    }

    out = goalformer(data_batch, training=False)
    import pdb

    pdb.set_trace()
    losses = goalformer.compute_losses(data_batch, training=False)
    for k, v in out.items():
        print(k, v.shape)
    for k, v in out.items():
        print(k, v.shape)
