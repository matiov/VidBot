from typing import Callable, List, Union

import cv2
from detectron2.layers import Conv2d, get_norm
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from vidbot.models.layers_2d import PositionEmbeddingSineImage, MLP, Up, ConvBlock, load_clip
from vidbot.models.transformer import TransformerEncoderLayer, Transformer, TransformerEncoder


class UnetPixelDecoder(nn.Module):
    def __init__(
        self,
        feature_keys: List[str],  # ["res1", "res2", "res3", "res4"]
        feature_channels: List[int],  # [64, 256, 512, 1024]
        output_channel=2,
        bilinear=True,
    ):
        super().__init__()
        feature_key2channel = {k: v for k, v in zip(feature_keys, feature_channels)}
        input_channel = feature_key2channel[feature_keys[-1]]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
        )

        self.up1 = Up(512 + feature_key2channel["res3"], 256, bilinear)

        self.up2 = Up(256 + feature_key2channel["res2"], 128, bilinear)

        self.up3 = Up(128 + feature_key2channel["res1"], 64, bilinear)

        self.up4 = Up(64, 64, bilinear)

        self.conv2 = ConvBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=True)
        self.conv3 = nn.Conv2d(64, output_channel, kernel_size=1)
        self.input_feature_key = feature_keys[-1]

    def forward(self, features):
        x = features[self.input_feature_key]
        x = self.conv1(x)
        x = self.up1(x, features["res3"])
        x = self.up2(x, features["res2"])
        x = self.up3(x, features["res1"])
        x = self.up4(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class BasePixelDecoder(nn.Module):
    def __init__(
        self,
        feature_keys: List[str],
        feature_channels: List[int],
        conv_dim: int,
        mask_dim: int,
        norm: Union[str, Callable] = "GN",
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()

        # input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = (
            feature_keys  # [k for k, v in input_shape]  # starting from "res2" to "res5"
        )
        # feature_channels = [v.channels for k, v in input_shape]
        self.input_shape = {k: v for k, v in zip(feature_keys, feature_channels)}

        lateral_convs = []
        output_convs = []

        use_bias = False  # norm == ""
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                output_norm = get_norm(norm, conv_dim)
                # output_norm = torch.nn.GroupNorm(32, conv_dim)
                output_conv = Conv2d(
                    in_channels,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(None)
                output_convs.append(output_conv)
            else:
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)
                # lateral_norm = torch.nn.GroupNorm(32, conv_dim)
                # output_norm = torch.nn.GroupNorm(32, conv_dim)

                lateral_conv = Conv2d(
                    in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
                )
                output_conv = Conv2d(
                    conv_dim,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                weight_init.c2_xavier_fill(lateral_conv)
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("adapter_{}".format(idx + 1), lateral_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(lateral_conv)
                output_convs.append(output_conv)

        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        self.mask_dim = mask_dim
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        weight_init.c2_xavier_fill(self.mask_features)

        self.maskformer_num_feature_levels = 3  # always use 3 scales

    def forward_features(self, features):
        multi_scale_features = []
        num_cur_levels = 0
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[::-1]):
            x = features[f]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            if lateral_conv is None:
                y = output_conv(x)
            else:
                cur_fpn = lateral_conv(x)
                # Following FPN implementation, we use nearest upsampling here
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)

            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(y)
                num_cur_levels += 1
        return self.mask_features(y), multi_scale_features

    def forward(self, features, targets=None):
        return self.forward_features(features)


class TransformerEncoderOnly(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        if mask is not None:
            mask = mask.flatten(1)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        return memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoderPixelDecoder(BasePixelDecoder):
    def __init__(
        self,
        feature_keys: List[str],
        feature_channels: List[int],
        transformer_dropout: float = 0.1,
        transformer_nheads: int = 8,
        transformer_dim_feedforward: int = 2048,
        transformer_enc_layers: int = 6,
        transformer_pre_norm: bool = False,
        conv_dim: int = 256,
        mask_dim: int = 256,
        norm: Union[str, Callable] = "GN",
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_enc_layers: number of transformer encoder layers
            transformer_pre_norm: whether to use pre-layernorm or not
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__(
            feature_keys, feature_channels, conv_dim=conv_dim, mask_dim=mask_dim, norm=norm
        )

        in_channels = feature_channels[len(self.in_features) - 1]
        self.input_proj = Conv2d(in_channels, conv_dim, kernel_size=1)
        weight_init.c2_xavier_fill(self.input_proj)

        self.transformer = TransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            normalize_before=transformer_pre_norm,
        )

        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSineImage(N_steps, normalize=True)

        # update layer
        use_bias = norm == ""
        output_norm = get_norm(norm, conv_dim)
        output_conv = Conv2d(
            conv_dim,
            conv_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
            norm=output_norm,
            activation=F.relu,
        )
        weight_init.c2_xavier_fill(output_conv)
        delattr(self, "layer_{}".format(len(self.in_features)))
        self.add_module("layer_{}".format(len(self.in_features)), output_conv)
        self.output_convs[0] = output_conv

    def forward_features(self, features):
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[::-1]):
            x = features[f]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            if lateral_conv is None:
                transformer = self.input_proj(x)
                pos = self.pe_layer(x)
                transformer = self.transformer(transformer, None, pos)
                y = output_conv(transformer)
                # save intermediate feature as input to Transformer decoder
                transformer_encoder_features = transformer
            else:
                cur_fpn = lateral_conv(x)
                # Following FPN implementation, we use nearest upsampling here
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)

        return self.mask_features(y), transformer_encoder_features

    def forward(self, features, targets=None):
        return self.forward_features(features)


class TransformerPredictor(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dropout: float,
        dim_feedforward: int,
        enc_layers: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        upsample_scale: int = 4,
    ):
        """
        Args:
            in_channels: channels of the input features
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dropout: dropout in Transformer
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            deep_supervision: whether to add supervision to every decoder layers
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSineImage(N_steps, normalize=True)

        transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=False,
        )

        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model

        if num_queries != -1:
            assert num_queries > 0
            self.query_embed = nn.Embedding(num_queries, hidden_dim)
        else:
            self.query_embed = None

        if in_channels != hidden_dim or enforce_input_project:
            self.input_proj = Conv2d(in_channels, hidden_dim, kernel_size=1)
            weight_init.c2_xavier_fill(self.input_proj)
        else:
            self.input_proj = nn.Sequential()

        # output FFNs
        self.out_embed = MLP(hidden_dim, mask_dim, [hidden_dim] * 3)
        self.upsample_scale = upsample_scale

    def forward(self, x, mask_features, query_features=None, upscale=True):
        batch_size = x.shape[0]
        pos = self.pe_layer(x)
        if query_features is None:
            query_features = self.query_embed.weight[None].repeat(
                batch_size, 1, 1
            )  # [bs, queries, embed]
        # import pdb; pdb.set_trace()
        # [l, bs, queries, embed]
        hs, memory = self.transformer(
            self.input_proj(x), None, query_features, pos
        )  # Low res discrete tokens
        out = torch.einsum("bqc,bchw->bqhw", self.out_embed(hs[-1]), mask_features)
        if upscale:
            out = F.interpolate(
                out, scale_factor=self.upsample_scale, mode="bilinear", align_corners=False
            )
        return out


if __name__ == "__main__":
    with torch.no_grad():
        # Reshape embeddings from flattened patches to patch height and width
        image = cv2.imread(
            "/home/wiss/chenh/hand_object_percept/egoprior-3D/assets/EPIC-KITCHENS/demo/5/rgb_frames/0000001040.jpg"
        )[:256, :448]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image).unsqueeze(0).float()
        h_in, w_in = image.shape[-2:]

        encoder, transform = load_clip()

        decoder = TransformerEncoderPixelDecoder(
            feature_keys=["res1", "res2", "res3", "res4", "res5"],
            feature_channels=[64, 256, 512, 1024, 2048],
            conv_dim=256,
            mask_dim=256,
        )
        predictor = TransformerPredictor(
            in_channels=256,
            hidden_dim=256,
            num_queries=2 * 5,
            nheads=8,
            dropout=0.1,
            dim_feedforward=2048,
            enc_layers=6,
            dec_layers=6,
            pre_norm=False,
            mask_dim=256,
            enforce_input_project=False,
        )
        unet_decoder = UnetPixelDecoder(
            feature_keys=["res1", "res2", "res3", "res4", "res5"],
            feature_channels=[64, 256, 512, 1024, 2048],
        )

        features = encoder(image)
        for k, v in features.items():
            print(k, v.shape)
        x = unet_decoder(features)

        import pdb

        pdb.set_trace()
