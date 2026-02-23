# Copyright (c) 2022-present, Kakao Brain Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from vidbot.models.residual_vq import ResidualVQ
from vidbot.models.layers_2d import Encoder, Decoder, ResnetBlock


class RQVAE(pl.LightningModule):
    def __init__(
        self,
        embed_dim=256,
        num_embed=32,
        num_vq=4,
        loss_type="l1",
        latent_loss_weight=5,
        ddconfig=None,
        checkpointing=False,
        **kwargs
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_vq = num_vq
        self.num_embed = num_embed
        self.ddconfig = ddconfig

        assert loss_type in ["mse", "l1"]

        # Encoder and Decoder
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        def set_checkpointing(m):
            if isinstance(m, ResnetBlock):
                m.checkpointing = checkpointing

        self.encoder.apply(set_checkpointing)
        self.decoder.apply(set_checkpointing)

        # Quantizer
        self.quantizer = ResidualVQ(
            dim=self.embed_dim,
            num_quantizers=self.num_vq,
            codebook_size=self.num_embed,
        )

        self.quant_conv = nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        self.loss_type = loss_type
        self.latent_loss_weight = latent_loss_weight

    @staticmethod
    def scale(x):
        # import pdb; pdb.set_trace()
        assert x.min() >= 0 and x.max() <= 1, "input should be in the range [0, 1]"
        return x * 2 - 1  # [-1, 1]

    @staticmethod
    def descale(x):
        x = x.clamp(-1, 1)  # [-1, 1]
        x = (x + 1) / 2  # [0, 1]
        return x

    def draw_logits_forward(self, encoding_logits):
        z_embed = self.quantizer.draw_logits_forward(encoding_logits)
        return z_embed

    def draw_code_forward(self, encoding_indices):
        with torch.no_grad():
            z_embed = self.quantizer.get_codes_from_indices(encoding_indices)
            z_embed = z_embed.sum(dim=0)
        return z_embed

    def quantize(self, z_e):
        batch, height, width, _ = z_e.shape
        z_e = z_e.view(batch, height * width, -1)
        z_q, code, quant_loss = self.quantizer(z_e)  # [B, H, W, C'], [B, H, W, num_vq]
        z_q = z_q.view(batch, height, width, -1)
        code = code.view(batch, height, width, -1)
        return z_q, code, quant_loss

    def forward(self, data_batch, key="vfd", return_rec=False):
        outputs = {}
        x = data_batch[key]
        x_inp = self.scale(x)
        z_e = self.encode(x_inp)
        z_q, code, _ = self.quantize(z_e)
        outputs.update({key + "_code": z_q, key + "_code_id": code})
        if return_rec:
            out = self.decode(z_q)
            out = self.descale(out)
            if key == "vfd":
                out = out.clamp(0, 1)
                out[:, :2] = out[:, :2] * 2 - 1  # [-1, 1]
                assert (
                    out[:, :2].min() >= -1 and out[:, :2].max() <= 1
                ), "VF output should be in the range [-1, 1]"
                assert (
                    out[:, 2].min() >= 0 and out[:, 2].max() <= 1
                ), "Invdepth factor output should be in the range [0, 1]"
            outputs.update({key + "_rec": out})
        return outputs

    def encode_decode(self, xs):
        z_e = self.encode(xs)
        z_q, code, quant_loss = self.quantize(z_e)
        out = self.decode(z_q)
        return out, code, quant_loss

    def encode(self, x):
        z_e = self.encoder(x)  # [B, C, H, W]
        z_e = self.quant_conv(z_e).permute(0, 2, 3, 1).contiguous()  # [B, H, W, C']
        return z_e

    def decode(self, z_q):
        z_q = z_q.permute(0, 3, 1, 2).contiguous()  # [B, C', H, W]
        z_q = self.post_quant_conv(z_q)  # [B, C, H, W]
        out = self.decoder(z_q)  # [B, C, H, W]
        return out

    def compute_losses(self, data_batch, key="vfd"):  # , quant_loss, code, xs=None, valid=False):
        x = data_batch[key]
        x_inp = self.scale(x)
        x_out, code, quant_loss = self.encode_decode(x_inp)
        x_rec = self.descale(x_out)
        loss_vq = quant_loss.sum()

        if self.loss_type == "mse":
            loss_recon = F.mse_loss(x_out, x_inp, reduction="mean")
        elif self.loss_type == "l1":
            loss_recon = F.l1_loss(x_out, x_inp, reduction="mean")
        else:
            raise ValueError("incompatible loss type")

        loss_total = loss_recon + self.latent_loss_weight * loss_vq

        return {
            "total_loss": loss_total,
            "recon_loss": loss_recon,
            "latent_loss": loss_vq,
        }


if __name__ == "__main__":
    ddconfig = {
        "ch": 128,
        "out_ch": 3,
        "ch_mult": [1, 1, 2, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_resolutions": (8,),
        "dropout": 0.0,
        "in_channels": 3,
        "resolution": [256, 448],
        "z_channels": 256,
        "double_z": False,
    }
    rqvae = RQVAE(
        embed_dim=256,
        num_embed=1024,
        num_vq=4,
        loss_type="mse",
        latent_loss_weight=5,
        checkpointing=True,
        ddconfig=ddconfig,
    )
    x = torch.rand(1, 3, 256, 448)
    data = {"vfd": x}
    out = rqvae(data, return_rec=True)
    for k, v in out.items():
        print(k, v.shape)
        if k == "vfd_code_id":
            print(v.max(), v.min())
