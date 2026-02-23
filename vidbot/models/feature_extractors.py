import math
from typing import List, Tuple
import types

import einops
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.ops import FeaturePyramidNetwork

from vidbot.models.layers_2d import (
    RelativeCrossAttentionModule,
    BackprojectDepth,
    load_clip,
    Project3D,
    Up,
    ConvBlock,
)
from vidbot.models.helpers import TSDFVolume, get_view_frustum
from vidbot.models.layers_3d import VoxelGridEncoder, RotaryPositionEncoding3D
from vidbot.models.preceiver import FeaturePreceiver
import vidbot.models.gcn3d as gcn3d  # type: ignore


#  code from https://github.com/f3rm/f3rm/blob/main/f3rm/features/dino/dino_vit_extractor.py
class ViTExtractor(nn.Module):
    """This class facilitates extraction of features, descriptors, and saliency maps from a ViT.

    We use the following notation in the documentation of the module's methods:
    B - batch size
    h - number of heads. usually takes place of the channel dimension in pytorch's convention BxCxHxW
    p - patch size of the ViT. either 8 or 16.
    t - number of tokens. equals the number of patches + 1, e.g. HW / p**2 + 1. Where H and W are the height and width
    of the input image.
    d - the embedding dimension in the ViT.
    """

    def __init__(
        self,
        model_type: str = "dino_vits8",
        stride: int = 4,
        model: nn.Module = None,
        load_size: int = 224,
    ):
        super(ViTExtractor, self).__init__()
        """
        :param model_type: A string specifying the type of model to extract from.
                          [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 |
                          vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]
        :param stride: stride of first convolution layer. small stride -> higher resolution.
        :param model: Optional parameter. The nn.Module to extract from instead of creating a new one in ViTExtractor.
                      should be compatible with model_type.
        """
        self.model_type = model_type
        if model is not None:
            self.model = model
        else:
            self.model = ViTExtractor.create_model(model_type)
        if stride is not None and not "dinov2_" in model_type:
            self.model = ViTExtractor.patch_vit_resolution(self.model, stride=stride)
        self.model.eval()
        self.p = self.model.patch_embed.patch_size
        self.stride = self.model.patch_embed.proj.stride

        self.mean = (0.485, 0.456, 0.406) if "dino" in self.model_type else (0.5, 0.5, 0.5)
        self.std = (0.229, 0.224, 0.225) if "dino" in self.model_type else (0.5, 0.5, 0.5)

        self._feats = []
        self.hook_handlers = []
        self.load_size = load_size
        self.num_patches = None

    @staticmethod
    def create_model(model_type: str):
        """
        :param model_type: a string specifying which model to load.
        [
        dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 |
        dinov2_vitb14 | dinov2_vitbs14 | dinov2_vitb14_reg | dinov2_vitbs14_reg |
        vit_small_patch8_224 |  vit_small_patch16_224 | vit_base_patch8_224 |  vit_base_patch16_224

        ]
        :return: the model
        """
        if "dino_" in model_type:
            model = torch.hub.load("facebookresearch/dino:main", model_type)

        elif "dinov2_" in model_type:
            model = torch.hub.load("facebookresearch/dinov2", model_type)

        else:  # model from timm -- load weights from timm to dino model (enables working on arbitrary size images).
            try:
                import timm
            except ImportError:
                raise ImportError("Please install timm: pip install timm")

            temp_model = timm.create_model(model_type, pretrained=True)
            model_type_dict = {
                "vit_small_patch16_224": "dino_vits16",
                "vit_small_patch8_224": "dino_vits8",
                "vit_base_patch16_224": "dino_vitb16",
                "vit_base_patch8_224": "dino_vitb8",
            }
            model = torch.hub.load("facebookresearch/dino:main", model_type_dict[model_type])
            temp_state_dict = temp_model.state_dict()
            try:
                del temp_state_dict["head.weight"]
                del temp_state_dict["head.bias"]
            except:
                pass
            model.load_state_dict(temp_state_dict)
        return model

    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):
        """
        Creates a method for position encoding interpolation.
        :param patch_size: patch size of the model.
        :param stride_hw: A tuple containing the new height and width stride respectively.
        :return: the interpolation method
        """

        def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int):
            npatch = x.shape[1] - 1
            N = self.pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:]
            dim = x.shape[-1]
            # compute number of tokens taking stride into account
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            assert (
                w0 * h0 == npatch
            ), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and 
                                            stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(
                    0, 3, 1, 2
                ),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode="bicubic",
                align_corners=False,
                recompute_scale_factor=False,
            )
            assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        return interpolate_pos_encoding

    @staticmethod
    def patch_vit_resolution(model: nn.Module, stride: int):
        """
        change resolution of model output by changing the stride of the patch extraction.
        :param model: the model to change resolution for.
        :param stride: the new stride parameter.
        :return: the adjusted model
        """
        patch_size = model.patch_embed.patch_size
        if stride == patch_size:  # nothing to do
            return model

        stride = torch.nn.modules.utils._pair(stride)
        assert all(
            [(patch_size // s_) * s_ == patch_size for s_ in stride]
        ), f"stride {stride} should divide patch_size {patch_size}"

        # fix the stride
        model.patch_embed.proj.stride = stride
        # fix the positional encoding code
        model.interpolate_pos_encoding = types.MethodType(
            ViTExtractor._fix_pos_enc(patch_size, stride), model
        )
        return model

    def preprocess(
        self,
        data_batch: torch.Tensor,
    ):
        if self.load_size is not None:
            data_batch = torch.nn.functional.interpolate(
                data_batch, size=self.load_size, mode="bicubic", align_corners=False
            )
        prep = transforms.Compose([transforms.Normalize(mean=self.mean, std=self.std)])
        data_batch = prep(data_batch)
        return data_batch

    def _get_hook(self, facet: str):
        """
        generate a hook method for a specific block and facet.
        """
        if facet in ["attn", "token"]:

            def _hook(model, input, output):
                self._feats.append(output)

            return _hook

        if facet == "query":
            facet_idx = 0
        elif facet == "key":
            facet_idx = 1
        elif facet == "value":
            facet_idx = 2
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        def _inner_hook(module, input, output):
            input = input[0]
            B, N, C = input.shape
            qkv = (
                module.qkv(input)
                .reshape(B, N, 3, module.num_heads, C // module.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            self._feats.append(qkv[facet_idx])  # Bxhxtxd

        return _inner_hook

    def _register_hooks(self, layers: List[int], facet: str):
        """
        register hook to extract features.
        :param layers: layers from which to extract features.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        """
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in layers:
                if facet == "token":
                    self.hook_handlers.append(block.register_forward_hook(self._get_hook(facet)))
                elif facet == "attn":
                    self.hook_handlers.append(
                        block.attn.attn_drop.register_forward_hook(self._get_hook(facet))
                    )
                elif facet in ["key", "query", "value"]:
                    self.hook_handlers.append(
                        block.attn.register_forward_hook(self._get_hook(facet))
                    )
                else:
                    raise TypeError(f"{facet} is not a supported facet.")

    def _unregister_hooks(self):
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    def _extract_features(self, batch: torch.Tensor, layers: List[int] = 11, facet: str = "key"):
        """
        extract features from the model
        :param batch: batch to extract features for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        :return : tensor of features.
                  if facet is 'key' | 'query' | 'value' has shape Bxhxtxd
                  if facet is 'attn' has shape Bxhxtxt
                  if facet is 'token' has shape Bxtxd
        """
        B, C, H, W = batch.shape
        self._feats = []
        self._register_hooks(layers, facet)
        _ = self.model(batch)
        self._unregister_hooks()
        self.load_size = (H, W)
        self.num_patches = (
            1 + (H - self.p) // self.stride[0],
            1 + (W - self.p) // self.stride[1],
        )
        return self._feats

    def extract_descriptors(
        self,
        batch: torch.Tensor,
        layer: int = 11,
        facet: str = "key",
        include_cls: bool = False,
    ):
        """
        extract descriptors from the model
        :param batch: batch to extract descriptors for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token']
        :param bin: apply log binning to the descriptor. default is False.
        :return: tensor of descriptors. Bx1xtxd' where d' is the dimension of the descriptors.
        """
        assert facet in [
            "key",
            "query",
            "value",
            "token",
        ], f"""{facet} is not a supported facet for descriptors. 
                                                             choose from ['key' | 'query' | 'value' | 'token'] """

        batch = self.preprocess(batch)
        self._extract_features(batch, [layer], facet)
        x = self._feats[0]
        if facet == "token":
            x.unsqueeze_(dim=1)  # Bx1xtxd
        if not include_cls:
            x = x[:, :, 1:, :]  # remove cls token
        else:
            assert (
                not bin
            ), "bin = True and include_cls = True are not supported together, set one of them False."
        desc = (
            x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1).unsqueeze(dim=1)
        )  # Bx1xtx(dxh)

        return desc

    def extract_saliency_maps(self, batch: torch.Tensor):
        """
        extract saliency maps. The saliency maps are extracted by averaging several attention heads from the last layer
        in of the CLS token. All values are then normalized to range between 0 and 1.
        :param batch: batch to extract saliency maps for. Has shape BxCxHxW.
        :return: a tensor of saliency maps. has shape Bxt-1
        """
        assert (
            self.model_type == "dino_vits8"
        ), f"saliency maps are supported only for dino_vits model_type."
        self._extract_features(batch, [11], "attn")
        head_idxs = [0, 2, 4, 5]
        curr_feats = self._feats[0]  # Bxhxtxt
        cls_attn_map = curr_feats[:, head_idxs, 0, 1:].mean(dim=1)  # Bx(t-1)
        temp_mins, temp_maxs = cls_attn_map.min(dim=1)[0], cls_attn_map.max(dim=1)[0]
        cls_attn_maps = (cls_attn_map - temp_mins) / (
            temp_maxs - temp_mins
        )  # normalize to range [0,1]
        return cls_attn_maps

    def forward(
        self,
        batch: torch.Tensor,
    ):
        batch = self.preprocess(batch)

        if self.model_type in [
            "dinov2_vitb14",
            "dinov2_vits14",
            "dinov2_vitb14_reg",
            "dinov2_vits14_reg",
        ]:
            batch = self.model.prepare_tokens_with_masks(batch, None)
        else:
            batch = self.model.prepare_tokens(batch)
        for blk in self.model.blocks:
            batch = blk(batch)
        offset = 1
        if self.model in ["dinov2_vitb14_reg", "dinov2_vits14_reg"]:
            offset += self.model.num_register_tokens
        elif self.model in ["simpool_vits16"]:
            offset += -1

        return batch[:, offset:]


class PointsMapFeatureExtractor(nn.Module):
    def __init__(
        self,
        input_image_shape,  # [H, W]
        embedding_dim=60,
        weight_tying=True,
        num_attn_heads=4,
        num_attn_layers=1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_image_shape = input_image_shape
        self.weight_tying = weight_tying
        self.backproject = BackprojectDepth(input_image_shape[0], input_image_shape[1])

        # Load pretrained backbone
        self.backbone, self.normalize = load_clip()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Load feature pyramid network
        self.feature_map_pyramid_keys = ["res1", "res2", "res3"]
        self.feature_pyramid = FeaturePyramidNetwork(
            [64, 256, 512, 1024, 2048], self.embedding_dim
        )

        # Load position encoding module
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)

        # Load relative cross attention module
        self.cross_attn_pyramid = nn.ModuleList()
        self.point_emb_pyramid = nn.ModuleList()
        if self.weight_tying:
            cross_attn_module = RelativeCrossAttentionModule(
                embedding_dim, num_attn_heads, num_attn_layers
            )
            point_emb = nn.Embedding(1, embedding_dim)
            for _ in range(len(self.feature_map_pyramid_keys)):
                self.cross_attn_pyramid.append(cross_attn_module)
                self.point_emb_pyramid.append(point_emb)
        else:
            for _ in range(len(self.feature_map_pyramid_keys)):
                cross_attn_module = RelativeCrossAttentionModule(
                    embedding_dim, num_attn_heads, num_attn_layers
                )
                self.cross_attn_pyramid.append(cross_attn_module)
                self.point_emb_pyramid.append(nn.Embedding(1, embedding_dim))

        # Load the feature fusion module
        self.last_layer = nn.Linear(
            embedding_dim * len(self.feature_map_pyramid_keys), embedding_dim
        )

    def compute_context_features(self, color, depth, intrinsics):
        color = self.normalize(color)
        color_features = self.backbone(color)
        color_features = self.feature_pyramid(color_features)
        color_features_pyramid = []

        # Backproject the points should have shape [B, 3, H, W]
        points = self.backproject(depth, intrinsics)  # [B, 3, H*W]
        points_map_pyramid = [
            points.view(-1, 3, self.input_image_shape[0], self.input_image_shape[1])
        ] * 3  # [B, 3, H, W]
        points_pe_pyramid = [
            self.relative_pe_layer(points.permute(0, 2, 1))
        ] * 3  # [B, H*W, 3] => [B, H*W, C, 2]

        for i, k in enumerate(self.feature_map_pyramid_keys):
            color_feature_i = color_features[k]
            color_feature_i = F.interpolate(
                color_feature_i, size=self.input_image_shape, mode="bilinear"
            )
            color_features_pyramid.append(color_feature_i)
        return color_features_pyramid, points_map_pyramid, points_pe_pyramid

    def forward(
        self,
        color_features_pyramid,
        points_map_pyramid,
        points_pe_pyramid,
        query_points,
        **kwargs,
    ):
        # batch_size = color.shape[0]
        batch_size, num_query_points, _ = query_points.shape
        # color_features_pyramid, points_map_pyramid, points_pe_pyramid = self.compute_color_features(color, depth)

        features = []
        for i, k in enumerate(self.feature_map_pyramid_keys):
            # Re-arrange to feature maps
            color_feature_i = color_features_pyramid[i]  # [B, C, H, W]
            points_map_i = points_map_pyramid[i]  # [B, 3, H, W]
            points_pe_i = points_pe_pyramid[i]  # [B, H*W, C, 2]

            map_feature_i = einops.rearrange(color_feature_i, "B C H W -> B (H W) C").permute(
                1, 0, 2
            )  # [H*W, B, C]
            map_pe_i = points_pe_i
            query_feature_i = (
                self.point_emb_pyramid[i]
                .weight.unsqueeze(0)
                .repeat(num_query_points, batch_size, 1)
            )  # [N, B, C]
            query_pe_i = self.relative_pe_layer(query_points)  # [B, N, 3] => [B, N, C, 2]

            # Pass through the relative cross attention module
            cross_attn_module = self.cross_attn_pyramid[i]
            feat = cross_attn_module(
                query=query_feature_i,
                value=map_feature_i,
                query_pos=query_pe_i,
                value_pos=map_pe_i,
            )
            features.append(feat[-1])

        features = torch.cat(features, dim=-1)  # [N, B, C*3]
        features = features.permute(1, 0, 2)  # [B, N, C*3]
        features = self.last_layer(features)
        return features


class MultiScaleImageFeatureExtractor(nn.Module):
    _RESNET_MEAN = [0.485, 0.456, 0.406]
    _RESNET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        modelname: str = "dino_vits16",
        freeze: bool = False,
        scale_factors: list = [1, 1 / 2, 1 / 3],
        embedding_dim: int = None,
    ):
        super().__init__()
        self.freeze = freeze
        self.scale_factors = scale_factors
        self.embedding_dim = embedding_dim

        if "res" in modelname:
            self._net = getattr(torchvision.models, modelname)(pretrained=True)
            self._output_dim = self._net.fc.weight.shape[1]
            self._net.fc = nn.Identity()
        elif "dinov2" in modelname:
            self._net = torch.hub.load("facebookresearch/dinov2", modelname)
            self._output_dim = self._net.norm.weight.shape[0]
        elif "dino" in modelname:
            self._net = torch.hub.load("facebookresearch/dino:main", modelname)
            self._output_dim = self._net.norm.weight.shape[0]
        else:
            raise ValueError(f"Unknown model name {modelname}")

        for name, value in (
            ("_resnet_mean", self._RESNET_MEAN),
            ("_resnet_std", self._RESNET_STD),
        ):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 3, 1, 1), persistent=False)

        if self.freeze:
            for param in self.parameters():
                param.requires_grad = False

        if self.embedding_dim is not None:
            self._last_layer = nn.Linear(self._output_dim, self.embedding_dim)
            self._output_dim = self.embedding_dim
        else:
            self._last_layer = nn.Identity()

    def get_output_dim(self):
        return self._output_dim

    def forward(self, image_rgb: torch.Tensor):
        img_normed = self._resnet_normalize_image(image_rgb)
        features = self._compute_multiscale_features(img_normed)
        return features

    def _resnet_normalize_image(self, img: torch.Tensor):
        return (img - self._resnet_mean) / self._resnet_std

    def _compute_multiscale_features(self, img_normed: torch.Tensor):
        multiscale_features = None

        if len(self.scale_factors) <= 0:
            raise ValueError(f"Wrong format of self.scale_factors: {self.scale_factors}")

        for scale_factor in self.scale_factors:
            if scale_factor == 1:
                inp = img_normed
            else:
                inp = self._resize_image(img_normed, scale_factor)

            if multiscale_features is None:
                multiscale_features = self._net(inp)
            else:
                multiscale_features += self._net(inp)

        averaged_features = multiscale_features / len(self.scale_factors)
        averaged_features = self._last_layer(averaged_features)
        return averaged_features

    @staticmethod
    def _resize_image(image: torch.Tensor, scale_factor: float):
        return nn.functional.interpolate(
            image, scale_factor=scale_factor, mode="bilinear", align_corners=False
        )


class MapGridDecoder(nn.Module):
    """
    Has 3 doubling up-samples.
    UNet part based on https://github.com/milesial/Pytorch-UNet/tree/master/unet
    """

    def __init__(
        self,
        input_shape,
        output_channel,
        encoder_channels,
        bilinear=True,
        batchnorm=True,
    ):
        super(MapGridDecoder, self).__init__()
        input_channel = input_shape[0]
        input_hw = np.array(input_shape[1:])
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
        )

        self.up1 = Up(512 + encoder_channels[-1], 256, bilinear)
        input_hw = input_hw * 2

        self.up2 = Up(256 + encoder_channels[-2], 128, bilinear)
        input_hw = input_hw * 2

        self.up3 = Up(128 + encoder_channels[-3], 64, bilinear)
        input_hw = input_hw * 2

        self.conv2 = ConvBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=batchnorm)
        self.conv3 = nn.Conv2d(64, output_channel, kernel_size=1)
        self.out_norm = nn.LayerNorm((output_channel, int(input_hw[0]), int(input_hw[1])))

    def forward(self, feat_to_decode: torch.Tensor, encoder_feats: List[torch.Tensor]):
        assert len(encoder_feats) >= 3
        x = self.conv1(feat_to_decode)
        x = self.up1(x, encoder_feats[-1])
        x = self.up2(x, encoder_feats[-2])
        x = self.up3(x, encoder_feats[-3])
        x = self.conv2(x)
        x = self.conv3(x)
        return self.out_norm(x)


class FeatureMapGridDecoder(nn.Module):
    """
    Has 3 doubling up-samples.
    UNet part based on https://github.com/milesial/Pytorch-UNet/tree/master/unet
    """

    def __init__(self, input_shape, output_channel, bilinear=True, batchnorm=True):
        super(FeatureMapGridDecoder, self).__init__()
        input_channel = input_shape[0]
        input_hw = np.array(input_shape[1:])
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
        )

        self.up1 = Up(512, 256, bilinear)  # [H//8, W//8]
        input_hw = input_hw * 2

        self.up2 = Up(256, 128, bilinear)  # [H//4, W//4]
        input_hw = input_hw * 2

        self.up3 = Up(128, 64, bilinear)  # [H//2, W//2]
        input_hw = input_hw * 2

        self.up4 = Up(64, 64, bilinear)  # [H//2, W//2]
        input_hw = input_hw * 2

        self.conv2 = ConvBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=batchnorm)
        self.conv3 = nn.Conv2d(64, output_channel, kernel_size=1)
        self.out_norm = nn.LayerNorm((output_channel, int(input_hw[0]), int(input_hw[1])))

    def forward(self, feat_to_decode):
        x = self.conv1(feat_to_decode)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.out_norm(x)


class TSDFMapFeatureExtractor(nn.Module):
    def __init__(
        self,
        input_image_shape,
        voxel_resolution=32,
        voxel_feature_dim=64,
        vlm_feature_attn_dim=256,
        # use_feature_decoder=True,
    ):
        super().__init__()
        self.input_image_shape = input_image_shape
        self.voxel_resolution = voxel_resolution
        self.embedding_dim = voxel_feature_dim
        self.backproject = BackprojectDepth(input_image_shape[0], input_image_shape[1])
        self.project_3d = Project3D()

        # Load pretrained backbone
        # self.vlm, self.vlm_transform = clip.load("ViT-B/16", jit=False)
        self.vlm, self.vlm_transform = load_clip()

        self.vlm.float()
        for p in self.vlm.parameters():
            p.requires_grad = False

        # Load 3D Unet
        self.tsdf_net = VoxelGridEncoder(self.voxel_resolution, c_dim=self.embedding_dim)

        self.feature_pyramid = FeaturePyramidNetwork([64, 256, 512, 1024, 2048], voxel_feature_dim)

        self.feature_map_pyramid_keys = ["res1", "res2", "res3"]

        # Cross Attention Layer
        # self.action_proj = nn.Linear(vlm_feature_attn_dim, self.embedding_dim, bias=True)
        self.vlm_preceiver_pyramid = nn.ModuleList()
        self.vlm_proj_pyramid = nn.ModuleList()
        vlm_preceiver = FeaturePreceiver(
            transition_dim=self.embedding_dim,
            condition_dim=vlm_feature_attn_dim,
            time_emb_dim=0,
        )
        vlm_proj = nn.Linear(vlm_preceiver.last_dim, self.embedding_dim, bias=True)
        for _ in range(len(self.feature_map_pyramid_keys)):

            self.vlm_preceiver_pyramid.append(vlm_preceiver)
            self.vlm_proj_pyramid.append(vlm_proj)
        # Feature projection layer
        proj_dim_in = voxel_feature_dim * (1 + len(self.feature_map_pyramid_keys))
        self.proj = nn.Linear(proj_dim_in, self.embedding_dim, bias=True)

    def compute_tsdf_volume(self, color, depth, intrinsics, verbose=False):
        cam_pose = np.eye(4)
        tsdf_grid_batch = []
        tsdf_bounds_batch = []
        tsdf_color_batch = []
        mesh_batch = []
        for i in range(len(depth)):
            d_np = depth[i].cpu().numpy()[0]
            c_np = color[i].cpu().numpy().transpose(1, 2, 0)  # [H, W, 3], requested by TSDFVolume
            K_np = intrinsics[i].cpu().numpy()
            view_frust_pts = get_view_frustum(d_np, K_np, cam_pose)
            vol_bnds = np.zeros((3, 2))
            vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1)).min()
            vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1)).max()
            tsdf = TSDFVolume(vol_bnds, voxel_dim=self.voxel_resolution)
            tsdf.integrate(c_np * 255, d_np, K_np, cam_pose)
            tsdf_grid = torch.from_numpy(tsdf.get_tsdf_volume())
            tsdf_grid_batch.append(tsdf_grid)
            tsdf_bounds_batch.append(torch.from_numpy(vol_bnds[0]))
            if verbose:
                mesh = tsdf.get_mesh()
                color_grid = torch.from_numpy(tsdf.get_color_volume()) / 255.0
                mesh_batch.append(mesh)
                tsdf_color_batch.append(color_grid)
        tsdf_bounds_batch = torch.stack(tsdf_bounds_batch, dim=0).to(depth.device).float()
        tsdf_grid_batch = torch.stack(tsdf_grid_batch, dim=0).to(depth.device).float()
        if verbose:
            tsdf_color_batch = torch.stack(tsdf_color_batch, dim=0).to(depth.device).float()
            return tsdf_grid_batch, tsdf_color_batch, tsdf_bounds_batch, mesh_batch
        return tsdf_grid_batch

    def compute_context_features(self, color, depth, intrinsics, tsdf=None, action_features=None):
        if tsdf is None:
            tsdf = self.compute_tsdf_volume(color, depth, intrinsics)

        h_in, w_in = color.shape[-2:]

        color = self.vlm_transform(color)
        color_features = self.vlm(color)  # [B, N, C]
        color_features = self.feature_pyramid(color_features)  # [B, N, C]

        # Action grounding
        if action_features is not None:
            for i, k in enumerate(self.feature_map_pyramid_keys):
                color_feature_i = color_features[k]  #
                h, w = color_feature_i.shape[-2:]
                color_feature_i = einops.rearrange(color_feature_i, "B C H W-> B (H W) C")
                color_feature_i = self.vlm_preceiver_pyramid[i](
                    color_feature_i, action_features[:, None]
                )
                color_feature_i = self.vlm_proj_pyramid[i](color_feature_i)
                color_feature_i = einops.rearrange(
                    color_feature_i, "B (H W) C -> B C H W", H=h, W=w
                )
                color_features[k] = color_feature_i

        color_features_pyramid = []
        for i, k in enumerate(self.feature_map_pyramid_keys):
            color_feature_i = color_features[k]
            color_feature_i = F.interpolate(color_feature_i, size=(h_in, w_in), mode="bilinear")
            color_features_pyramid.append(color_feature_i)
        points_map_pyramid = [tsdf] * len(color_features_pyramid)  # [B, D, H, W]
        points_pe_pyramid = [self.tsdf_net(tsdf)] * len(color_features_pyramid)  # [B, P, D, H, W]

        return color_features_pyramid, points_map_pyramid, points_pe_pyramid

    @staticmethod
    def interpolate_voxel_grid_features(voxel_grid, query_points, voxel_bounds):
        """
        Parameters
        ----------
        voxel_grid : torch.Tensor
            with shape [B, C, D, H, W]
        query_points : torch.Tensor
            _with shape [B, N, 3]
        voxel_bounds: torch.Tensor
            _with shape [B, 2]
        """
        voxel_bounds = voxel_bounds.unsqueeze(-1).repeat(1, 1, 3)  # [B, 2, 3]
        query_points = (query_points - voxel_bounds[:, 0:1]) / (
            voxel_bounds[:, 1:2] - voxel_bounds[:, 0:1]
        )
        query_grids = query_points * 2 - 1  # Normalize the query points from [0, 1] to [-1, 1]
        query_grids = query_grids[..., [2, 1, 0]]  # Convert to the voxel grid coordinate system
        query_grids = query_grids[:, :, None, None]  # [B, N, 1, 1, 3]
        query_features = F.grid_sample(
            voxel_grid, query_grids, mode="bilinear", align_corners=True
        )  # [B, C, N, 1, 1]
        query_features = query_features.squeeze(-1).squeeze(-1)  # [B, C, N]
        return query_features

    def interpolate_image_grid_features(self, image_grid, query_points, intrinsics):
        """
        Parameters
        ----------
        image_grid : torch.Tensor
            with shape [B, C, H, W]
        query_points : torch.Tensor
            _with shape [B, N, 3]
        """
        batch_size, _, height, width = image_grid.shape
        query_grids = self.project_3d(query_points, intrinsics)  # [B, 2, N]
        query_grids[:, 0] = (query_grids[:, 0] / (width - 1)) * 2 - 1
        query_grids[:, 1] = (query_grids[:, 1] / (height - 1)) * 2 - 1
        query_grids = query_grids.permute(0, 2, 1)[:, :, None]  # [B, N, 1, 2]
        query_featurs = F.grid_sample(
            image_grid, query_grids, mode="bilinear", align_corners=True
        )  # [B, C, N, 1]
        query_featurs = query_featurs.squeeze(-1)
        return query_featurs

    def forward(
        self,
        color_features_pyramid,
        points_map_pyramid,
        points_pe_pyramid,
        query_points,
        intrinsics,
        voxel_bounds,
        **kwargs,
    ):
        """_summary_

        Parameters
        ----------
        color_features_pyramid :  list of torch.Tensor
            with shape [[B, C, H, W]]
        points_map_pyramid : list of torch.Tensor for TSDF volume
            [[B, D, H, W]]
        points_pe_pyramid :  list of torch.Tensor for TSDF volume feature
            [[B, P, D, H, W]]
        query_points : query points
            [B, N, 3]
        intrinsics : torch.Tensor or np.ndarray
            [3, 3]
        voxel_bounds : _type_
            [B, 2]

        Returns
        -------
        torch.Tensor
            shape of [B, N, C*4]
        """
        assert len(color_features_pyramid) == len(points_map_pyramid)
        assert len(color_features_pyramid) == len(points_pe_pyramid)
        batch_size, num_query_points, _ = query_points.shape
        features = []

        for i in range(len(color_features_pyramid)):
            # Re-arrange to feature maps
            color_feature_i = color_features_pyramid[i]  # [B, C, H, W]
            points_pe_i = points_pe_pyramid[i]  # [B, P, D, H, W]
            points_map_i = points_map_pyramid[i][:, None]  # [B, 1, D, H, W]

            if i == 0:
                # Interpolate the voxel grid features
                feat_occ = self.interpolate_voxel_grid_features(
                    points_map_i, query_points, voxel_bounds
                )

                # Interpolate the voxel grid features
                feat_3d = self.interpolate_voxel_grid_features(
                    points_pe_i, query_points, voxel_bounds
                )
                features.append(feat_3d)  # [B, C, N]

            # Interpolate the 2D feature maps
            feat_2d = self.interpolate_image_grid_features(
                color_feature_i, query_points, intrinsics
            )
            features.append(feat_2d)  # [B, C, N]
        features = torch.cat(features, dim=1).permute(0, 2, 1)  # [B, N, C*3]
        features = self.proj(features)  # [B, N, C]
        return features


class TSDFMapGeometryExtractor(nn.Module):
    def __init__(
        self,
        input_image_shape,
        voxel_resolution=64,
        voxel_feature_dim=64,
    ):
        super().__init__()
        self.input_image_shape = input_image_shape
        self.voxel_resolution = voxel_resolution
        self.embedding_dim = voxel_feature_dim
        self.backproject = BackprojectDepth(input_image_shape[0], input_image_shape[1])
        self.project_3d = Project3D()

        # Load 3D Unet
        self.tsdf_net = VoxelGridEncoder(self.voxel_resolution, c_dim=self.embedding_dim)

    def compute_tsdf_volume(self, color, depth, intrinsics, verbose=False):
        cam_pose = np.eye(4)
        tsdf_grid_batch = []
        tsdf_bounds_batch = []
        tsdf_color_batch = []
        mesh_batch = []
        for i in range(len(depth)):
            d_np = depth[i].cpu().numpy()[0]
            c_np = color[i].cpu().numpy().transpose(1, 2, 0)  # [H, W, 3], requested by TSDFVolume
            K_np = intrinsics[i].cpu().numpy()
            view_frust_pts = get_view_frustum(d_np, K_np, cam_pose)
            vol_bnds = np.zeros((3, 2))
            vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1)).min()
            vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1)).max()
            tsdf = TSDFVolume(vol_bnds, voxel_dim=self.voxel_resolution)
            tsdf.integrate(c_np * 255, d_np, K_np, cam_pose)
            tsdf_grid = torch.from_numpy(tsdf.get_tsdf_volume())
            tsdf_grid_batch.append(tsdf_grid)
            tsdf_bounds_batch.append(torch.from_numpy(vol_bnds[0]))
            if verbose:
                mesh = tsdf.get_mesh()
                color_grid = torch.from_numpy(tsdf.get_color_volume()) / 255.0
                mesh_batch.append(mesh)
                tsdf_color_batch.append(color_grid)
        tsdf_bounds_batch = torch.stack(tsdf_bounds_batch, dim=0).to(depth.device).float()
        tsdf_grid_batch = torch.stack(tsdf_grid_batch, dim=0).to(depth.device).float()
        if verbose:
            tsdf_color_batch = torch.stack(tsdf_color_batch, dim=0).to(depth.device).float()
            return tsdf_grid_batch, tsdf_color_batch, tsdf_bounds_batch, mesh_batch
        return tsdf_grid_batch

    def compute_context_features(self, color, depth, intrinsics, tsdf=None, action_featurs=None):
        if tsdf is None:
            tsdf = self.compute_tsdf_volume(color, depth, intrinsics)

        color_features_pyramid = [None]  # [B, C, H, W]
        points_map_pyramid = [tsdf]  # [B, D, H, W]
        points_pe_pyramid = [self.tsdf_net(tsdf)]  # [B, P, D, H, W]

        return color_features_pyramid, points_map_pyramid, points_pe_pyramid

    @staticmethod
    def interpolate_voxel_grid_features(voxel_grid, query_points, voxel_bounds):
        """
        Parameters
        ----------
        voxel_grid : torch.Tensor
            with shape [B, C, D, H, W]
        query_points : torch.Tensor
            _with shape [B, N, 3]
        voxel_bounds: torch.Tensor
            _with shape [B, 2]
        """
        voxel_bounds = voxel_bounds.unsqueeze(-1).repeat(1, 1, 3)  # [B, 2, 3]
        query_points = (query_points - voxel_bounds[:, 0:1]) / (
            voxel_bounds[:, 1:2] - voxel_bounds[:, 0:1]
        )
        query_grids = query_points * 2 - 1  # Normalize the query points from [0, 1] to [-1, 1]
        query_grids = query_grids[..., [2, 1, 0]]  # Convert to the voxel grid coordinate system
        query_grids = query_grids[:, :, None, None]  # [B, N, 1, 1, 3]
        query_features = F.grid_sample(
            voxel_grid, query_grids, mode="bilinear", align_corners=True
        )  # [B, C, N, 1, 1]
        query_features = query_features.squeeze(-1).squeeze(-1)  # [B, C, N]
        return query_features

    def interpolate_image_grid_features(self, image_grid, query_points, intrinsics):
        """
        Parameters
        ----------
        image_grid : torch.Tensor
            with shape [B, C, H, W]
        query_points : torch.Tensor
            _with shape [B, N, 3]
        """
        batch_size, _, height, width = image_grid.shape
        query_grids = self.project_3d(query_points, intrinsics)  # [B, 2, N]
        query_grids[:, 0] = (query_grids[:, 0] / (width - 1)) * 2 - 1
        query_grids[:, 1] = (query_grids[:, 1] / (height - 1)) * 2 - 1
        query_grids = query_grids.permute(0, 2, 1)[:, :, None]  # [B, N, 1, 2]
        query_featurs = F.grid_sample(
            image_grid, query_grids, mode="bilinear", align_corners=True
        )  # [B, C, N, 1]
        query_featurs = query_featurs.squeeze(-1)
        return query_featurs

    def forward(
        self,
        color_features_pyramid,
        points_map_pyramid,
        points_pe_pyramid,
        query_points,
        intrinsics,
        voxel_bounds,
        **kwargs,
    ):
        """_summary_

        Parameters
        ----------
        color_features_pyramid :  list of torch.Tensor
            with shape [[B, C, H, W]]
        points_map_pyramid : list of torch.Tensor for TSDF volume
            [[B, D, H, W]]
        points_pe_pyramid :  list of torch.Tensor for TSDF volume feature
            [[B, P, D, H, W]]
        query_points : query points
            [B, N, 3]
        intrinsics : torch.Tensor or np.ndarray
            [3, 3]
        voxel_bounds : _type_
            [B, 2]

        Returns
        -------
        torch.Tensor
            shape of [B, N, C*4]
        """
        assert len(color_features_pyramid) == len(points_map_pyramid)
        assert len(color_features_pyramid) == len(points_pe_pyramid)
        batch_size, num_query_points, _ = query_points.shape
        features = []

        for i in range(len(color_features_pyramid)):
            # Re-arrange to feature maps
            color_feature_i = color_features_pyramid[i]  # [B, C, H, W]
            points_pe_i = points_pe_pyramid[i]  # [B, P, D, H, W]
            points_map_i = points_map_pyramid[i][:, None]  # [B, 1, D, H, W]

            if i == 0:
                # Interpolate the voxel grid features
                feat_occ = self.interpolate_voxel_grid_features(
                    points_map_i, query_points, voxel_bounds
                )

                # Interpolate the voxel grid features
                feat_3d = self.interpolate_voxel_grid_features(
                    points_pe_i, query_points, voxel_bounds
                )
                features.append(feat_3d)  # [B, C, N]

        features = torch.cat(features, dim=1).permute(0, 2, 1)  # [B, N, C*3]
        return features


class PointCloudFeatureExtractor(nn.Module):
    def __init__(self, support_num: int = 1, neighbor_num: int = 50, embedding_dim: int = 256):
        super().__init__()
        self.neighbor_num = neighbor_num

        self.conv_0 = gcn3d.Conv_surface(kernel_num=32, support_num=support_num)
        self.conv_1 = gcn3d.Conv_layer(32, 64, support_num=support_num)
        self.pool_1 = gcn3d.Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_2 = gcn3d.Conv_layer(64, 128, support_num=support_num)
        self.conv_3 = gcn3d.Conv_layer(128, 256, support_num=support_num)
        self.pool_2 = gcn3d.Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_4 = gcn3d.Conv_layer(256, 1024, support_num=support_num)
        self.fc = nn.Linear(1024, embedding_dim)

    def forward(self, vertices):
        bs, vertice_num, _ = vertices.size()

        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)
        fm_0 = self.conv_0(neighbor_index, vertices)
        fm_0 = F.relu(fm_0, inplace=True)
        fm_1 = self.conv_1(neighbor_index, vertices, fm_0)
        fm_1 = F.relu(fm_1, inplace=True)
        vertices, fm_1 = self.pool_1(vertices, fm_1)
        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)

        fm_2 = self.conv_2(neighbor_index, vertices, fm_1)
        fm_2 = F.relu(fm_2, inplace=True)
        fm_3 = self.conv_3(neighbor_index, vertices, fm_2)
        fm_3 = F.relu(fm_3, inplace=True)
        vertices, fm_3 = self.pool_2(vertices, fm_3)
        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)

        fm_4 = self.conv_4(neighbor_index, vertices, fm_3)
        feature_global = fm_4.max(1)[0]
        feature_global = self.fc(feature_global)
        return feature_global


if __name__ == "__main__":
    decoder = FeatureMapGridDecoder(input_shape=(256, 16, 28), output_channel=256)
    feat = torch.rand(6, 256, 16, 28)
    h = decoder(feat)
    print(h.shape)
