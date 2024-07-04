# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.model import ExponentialMovingAverage

from mmselfsup.registry import MODELS
from mmselfsup.structures import SelfSupDataSample
from mmselfsup.utils import (batch_shuffle_ddp, batch_unshuffle_ddp,
                             concat_all_gather)
from .base import BaseModel

import torch.nn.functional as F

@MODELS.register_module()
class MoCo(BaseModel):


    def __init__(self,
                 backbone: dict,
                 neck: dict,
                 head: dict,
                 queue_len: int = 65536,
                 feat_dim: int = 128,
                 momentum: float = 0.999,
                 pretrained: Optional[str] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            pretrained=pretrained,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        # create momentum model
        # self.encoder_k = ExponentialMovingAverage(
        #     nn.Sequential(self.backbone, self.neck), 1 - momentum)
        self.backbone_k = ExponentialMovingAverage(self.backbone, 1 - momentum)    ###
        self.neck_k = ExponentialMovingAverage(self.neck, 1 - momentum)            ###
        

        # create the queue
        self.queue_len = queue_len
        self.register_buffer('queue', torch.randn(feat_dim, queue_len))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))


        ###
        self.in_channels = [256, 512, 1024, 2048]
        self.out_channels = 512
        self.convs = nn.ModuleList()
        for i in range(4):
            self.convs.append(
                Conv2dReLU(
                    in_channels=self.in_channels[i],
                    out_channels=self.out_channels,
                    kernel_size=1
                )
            )

        self.ffl = FocalFrequencyLoss(loss_weight=1.0, alpha=1.0, ave_spectrum=True, log_matrix=True, batch_matrix=True)           ###
 
        # self.fusion_conv = Conv2dReLU(
        #     in_channels=
        # )

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        """Update queue."""
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr

    def extract_feat(self, inputs: List[torch.Tensor],
                     **kwarg) -> Tuple[torch.Tensor]:
        """Function to extract features from backbone.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Tuple[torch.Tensor]: Backbone outputs.
        """
        x = self.backbone(inputs[0])
        return x

    def loss(self, inputs: List[torch.Tensor],
             data_samples: List[SelfSupDataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        im_q = inputs[0]
        im_k = inputs[1]


        ####
        outs = []
        inputs_q = self.backbone(im_q)
        for idx in range(len(inputs_q)):
            x = inputs_q[idx]
            temp = self.convs[idx](x)
            if idx < 3:
                outs.append(F.interpolate(
                    input=temp,
                    size=(7, 7),
                    mode='bilinear'
                ))
            if idx == 3:
                outs.append(temp)
        
        out = torch.cat(outs, dim=1)
        out_q = (out,)



        q_dense, q_grid, q2_dense = self.neck_dense(out_q)  # queries: NxC; NxCxS^2
        out_q0 = out_q[0]
        out_q0 = out_q0.view(out_q0.size(0), out_q0.size(1), -1)

        q_dense = nn.functional.normalize(q_dense, dim=1)
        q2_dense = nn.functional.normalize(q2_dense, dim=1)
        q_grid = nn.functional.normalize(q_grid, dim=1)
        out_q0 = nn.functional.normalize(out_q0, dim=1)

        


        # compute query features from encoder_q
        q = self.neck(out_q)[0]  # queries: NxC
        q = nn.functional.normalize(q, dim=1)



        # compute key features
        with torch.no_grad():  # no gradient to keys
            # update the key encoder
            # self.encoder_k.update_parameters(
            #     nn.Sequential(self.backbone, self.neck))
            self.backbone_k.update_parameters(self.backbone)            ###
            self.neck_k.update_parameters(self.neck)                    ###

            # shuffle for making use of BN
            im_k, idx_unshuffle = batch_shuffle_ddp(im_k)

            # k = self.encoder_k(im_k)[0]  # keys: NxC

            outs_k = []
            inputs_k = self.backbone_k(im_k)                                       ###
            for idx in range(len(inputs_k)):
                x = inputs_k[idx]
                temp = self.convs[idx](x)
                if idx < 3:
                    outs_k.append(F.interpolate(
                        input=temp,
                        size=(7, 7),
                        mode='bilinear'
                    ))
                if idx == 3:
                    outs_k.append(temp)

            out_ = torch.cat(outs_k, dim=1)


            out_k = (out_,)

            k_dense, k_grid, k2_dense = self.neck_dense_k(out_k)  # keys: NxC; NxCxS^2
            out_k0 = out_k[0]
            out_k0 = out_k0.view(out_k0.size(0), out_k0.size(1), -1)

            k_dense = nn.functional.normalize(k_dense, dim=1)
            k2_dense = nn.functional.normalize(k2_dense, dim=1)
            k_grid = nn.functional.normalize(k_grid, dim=1)
            out_k0 = nn.functional.normalize(out_k0, dim=1)

            k_dense = batch_unshuffle_ddp(k_dense, idx_unshuffle)
            k2_dense = batch_unshuffle_ddp(k2_dense, idx_unshuffle)
            k_grid = batch_unshuffle_ddp(k_grid, idx_unshuffle)
            out_k0 = batch_unshuffle_ddp(out_k0, idx_unshuffle)
            
            k = self.neck_k(out_k)[0]                        ###

            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = batch_unshuffle_ddp(k, idx_unshuffle)


        # LF
        loss_ffl = self.ffl(out_q[0], out_k[0])


        backbone_sim_matrix = torch.matmul(out_q0.permute(0, 2, 1), k_b)
        densecl_sim_ind = backbone_sim_matrix.max(dim=2)[1]  # NxS^2

        indexed_k_grid = torch.gather(k_grid, 2,
                                      densecl_sim_ind.unsqueeze(1).expand(
                                          -1, k_grid.size(1), -1))  # NxCxS^2
        densecl_sim_q = (q_grid * indexed_k_grid).sum(1)  # NxS^2

        # dense positive logits: NS^2X1
        l_pos_dense = densecl_sim_q.view(-1).unsqueeze(-1)

        q_grid = q_grid.permute(0, 2, 1)
        q_grid = q_grid.reshape(-1, q_grid.size(2))
        # dense negative logits: NS^2xK
        l_neg_dense = torch.einsum(
            'nc,ck->nk', [q_grid, self.queue2.clone().detach()])


        # LP
        loss_dense = self.head(l_pos_dense, l_neg_dense)

        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # LI
        loss = self.head(l_pos, l_neg)
        # update the queue
        self._dequeue_and_enqueue(k)

        loss = (loss + loss_ffl) * 0.5 + loss_dense * 0.5

        losses = dict(loss=loss)
        return losses



class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class FocalFrequencyLoss(nn.Module):
    r"""Implements of focal frequency loss

    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for
            flexibility. Default: 1.0
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm.
            Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using
            batch-based statistics. Default: False
    """

    def __init__(self,
                 loss_weight=1.0,
                 alpha=1.0,
                 ave_spectrum=False,
                 log_matrix=False,
                 batch_matrix=False):

        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def loss_formulation(self, f_pred, f_targ, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            weight_matrix = matrix.detach()  # predefined
        else:
            # if the matrix is calculated online: continuous, dynamic,
            #   based on current Euclidean distance
            matrix_tmp = (f_pred - f_targ) ** 2  # loss越大，loss权重越大
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = \
                    matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
                'The values of spectrum weight matrix should be in the range [0, 1], '
                'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (f_pred - f_targ) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]
        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance

        return loss.mean()

    def forward(self, pred, target, matrix=None, **kwargs):
        r"""Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        f_pred = torch.fft.fft2(pred, dim=(2, 3), norm='ortho')
        f_targ = torch.fft.fft2(target, dim=(2, 3), norm='ortho')
        f_pred = torch.stack([f_pred.real, f_pred.imag], -1)
        f_targ = torch.stack([f_targ.real, f_targ.imag], -1)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            f_pred = torch.mean(f_pred, 0, keepdim=True)
            f_targ = torch.mean(f_targ, 0, keepdim=True)
        loss = self.loss_formulation(f_pred, f_targ, matrix) * self.loss_weight

        return loss