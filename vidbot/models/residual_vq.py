from functools import partial
from random import randrange

from einops import rearrange, repeat, pack, unpack
import torch
from torch import nn
import torch.nn.functional as F

from vidbot.models.helpers import exists, default, round_up_multiple
from vidbot.models.vector_quantize import VectorQuantize


class ResidualVQ(nn.Module):
    """from https://github.com/jayLEE0301/vq_bet_official/blob/main/vector_quantize_pytorch/residual_vq.py"""

    def __init__(
        self,
        *,
        dim,
        num_quantizers,  # depth of the quantizers
        codebook_dim=None,
        shared_codebook=False,
        heads=1,
        quantize_dropout=False,
        quantize_dropout_cutoff_index=0,
        quantize_dropout_multiple_of=1,
        accept_image_fmap=False,
        **kwargs
    ):
        super().__init__()
        assert heads == 1, "residual vq is not compatible with multi-headed codes"
        codebook_dim = default(codebook_dim, dim)
        codebook_input_dim = codebook_dim * heads

        requires_projection = codebook_input_dim != dim
        self.project_in = (
            nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        )
        self.project_out = (
            nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()
        )

        self.num_quantizers = num_quantizers

        self.accept_image_fmap = accept_image_fmap
        # Initialize D layers of vector quantizers
        # Note the first layer is to quantize feature, and the rest are to quantize residual
        self.layers = nn.ModuleList(
            [
                VectorQuantize(
                    dim=codebook_dim,
                    codebook_dim=codebook_dim,
                    accept_image_fmap=accept_image_fmap,
                    **kwargs
                )
                for _ in range(num_quantizers)
            ]
        )

        self.quantize_dropout = quantize_dropout and num_quantizers > 1

        assert quantize_dropout_cutoff_index >= 0

        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = quantize_dropout_multiple_of  # encodec paper proposes structured dropout, believe this was set to 4

        if not shared_codebook:
            return

        # share codebook across all layers
        first_vq, *rest_vq = self.layers
        codebook = first_vq._codebook
        for vq in rest_vq:
            vq._codebook = codebook

    @property
    def codebooks(self):
        codebooks = [
            layer._codebook.embed for layer in self.layers
        ]  # [1, num_codes, codebook_dim]
        codebooks = torch.stack(codebooks, dim=0)  # [num_vq, 1, num_codes, codebook_dim]
        codebooks = rearrange(codebooks, "q 1 c d -> q c d")  # [num_vq, num_codes, codebook_dim]
        return codebooks

    def get_codes_from_indices(self, indices):
        """_summary_

        Parameters
        ----------
        indices : [B, hw, num_vq]
            _description_

        """
        batch, quantize_dim = indices.shape[0], indices.shape[-1]

        # may also receive indices in the shape of 'b h w q' (accept_image_fmap)

        indices, ps = pack([indices], "b * q")  # ps = [hw]

        # because of quantize dropout, one can pass in indices that are coarse
        # and the network should be able to reconstruct

        if quantize_dim < self.num_quantizers:
            assert (
                self.quantize_dropout > 0.0
            ), "quantize dropout must be greater than 0 if you wish to reconstruct from a signal with less fine quantizations"
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value=-1)

        # get ready for gathering

        codebooks = repeat(
            self.codebooks, "q c d -> q b c d", b=batch
        )  # [num_vq, batch, num_codes, codebook_dim]
        gather_indices = repeat(
            indices, "b n q -> q b n d", d=codebooks.shape[-1]
        )  # [num_vq, batch, hw, codebook_dim]

        # take care of quantizer dropout

        mask = gather_indices == -1.0
        gather_indices = gather_indices.masked_fill(
            mask, 0
        )  # have it fetch a dummy code to be masked out later

        all_codes = codebooks.gather(
            2, gather_indices
        )  # gather all codes, [num_vq, batch, num_codes, codebook_dim] => [num_vq, batch, hw, codebook_dim]

        # mask out any codes that were dropout-ed

        all_codes = all_codes.masked_fill(mask, 0.0)

        # if (accept_image_fmap = True) then return shape (quantize, batch, height, width, dimension)

        (all_codes,) = unpack(all_codes, ps, "q b * d")  # [num_vq, batch, hw, codebook_dim]
        return all_codes

    def draw_logits_forward(self, encoding_logits):
        # encoding_indices : dim1 = batch_size  dim2 = 4 (number of groups) dim3 = vq dict size (header)
        encoding_logits = encoding_logits  # [batch, num_vq, num_codes]
        bs = encoding_logits.shape[0]
        quantized = torch.zeros((bs, self.codebooks.shape[-1]))  # [batch, codebook_dim]
        # codebooks = [num_vq, num_codes, codebook_dim]
        for q in range(encoding_logits.shape[1]):
            quantized += torch.matmul(encoding_logits[:, q], self.codebooks[q])
            # [batch, num_codes] @ [num_codes, codebook_dim] = [batch, codebook_dim]

        return quantized

    def forward(self, x, indices=None, return_all_codes=False, sample_codebook_temp=None):
        num_quant, quant_dropout_multiple_of, return_loss, device = (
            self.num_quantizers,
            self.quantize_dropout_multiple_of,
            exists(indices),
            x.device,
        )

        x = self.project_in(x)

        assert not (self.accept_image_fmap and exists(indices))

        quantized_out = 0.0
        residual = x

        all_losses = []
        all_indices = []

        if return_loss:
            assert not torch.any(
                indices == -1
            ), "some of the residual vq indices were dropped out. please use indices derived when the module is in eval mode to derive cross entropy loss"
            ce_losses = []

        should_quantize_dropout = self.training and self.quantize_dropout and not return_loss

        # sample a layer index at which to dropout further residual quantization
        # also prepare null indices and loss

        if should_quantize_dropout:
            rand_quantize_dropout_index = randrange(self.quantize_dropout_cutoff_index, num_quant)

            if quant_dropout_multiple_of != 1:
                rand_quantize_dropout_index = (
                    round_up_multiple(rand_quantize_dropout_index + 1, quant_dropout_multiple_of)
                    - 1
                )

            null_indices_shape = (
                (x.shape[0], *x.shape[-2:]) if self.accept_image_fmap else tuple(x.shape[:2])
            )
            null_indices = torch.full(null_indices_shape, -1.0, device=device, dtype=torch.long)
            null_loss = torch.full((1,), 0.0, device=device, dtype=x.dtype)

        # go through the layers

        for quantizer_index, layer in enumerate(self.layers):

            if should_quantize_dropout and quantizer_index > rand_quantize_dropout_index:
                all_indices.append(null_indices)
                all_losses.append(null_loss)
                continue

            layer_indices = None
            if return_loss:
                layer_indices = indices[..., quantizer_index]

            quantized, *rest = layer(
                residual,
                indices=layer_indices,
                sample_codebook_temp=sample_codebook_temp,
            )

            residual = residual - quantized.detach()  # e.g., r0 = x0, r1 = r0 - q(r0)
            quantized_out = (
                quantized_out + quantized
            )  # e.g., q0 = q(r0), q1 = q0 + q(r1), only the first term is quantized feature, the rest are n-order residuals

            if return_loss:
                ce_loss = rest[0]
                ce_losses.append(ce_loss)
                continue

            embed_indices, loss = rest

            all_indices.append(embed_indices)
            all_losses.append(loss)

        # project out, if needed

        quantized_out = self.project_out(quantized_out)

        # whether to early return the cross entropy loss

        if return_loss:
            return quantized_out, sum(ce_losses)

        # stack all losses and indices
        all_losses, all_indices = map(partial(torch.stack, dim=-1), (all_losses, all_indices))

        ret = (quantized_out, all_indices, all_losses)

        if return_all_codes:
            # whether to return all codes from all codebooks across layers
            all_codes = self.get_codes_from_indices(
                all_indices
            )  # [num_vq, batch, hw, codebook_dim]

            # will return all codes in shape (quantizer, batch, sequence length, codebook dimension)
            ret = (*ret, all_codes)

        return ret


if __name__ == "__main__":
    residual_vq = ResidualVQ(dim=256, num_quantizers=4, codebook_size=32)
    x = torch.randn(1, 20, 30, 256)  # (batch, sequence length, dimension)
    quantized, indices, commit_loss = residual_vq(
        x,
    )
    print(quantized.shape, indices.shape, commit_loss.shape)
