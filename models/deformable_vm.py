# ------------------------------------------------------------------------
# VMFormer Transformer part.
# ------------------------------------------------------------------------
# Modified from SeqFormer (https://github.com/wjf5203/SeqFormer)
# Copyright (c) 2021 Junfeng Wu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn

class DeformableVMFormer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False, 
                 query_temporal=None,
                 num_frames=1, num_feature_levels=4, 
                 dec_n_points=4,  enc_n_points=4,
                 ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_feature_levels = num_feature_levels
        

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels , 
                                                          nhead, enc_n_points)

        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels , 
                                                          nhead, dec_n_points, query_temporal)

        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        assert  query_embed is not None
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, nf, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(3).transpose(2, 3)
            mask = mask.flatten(2)
            pos_embed = pos_embed.flatten(3).transpose(2, 3)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, 1, -1)

            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        
        src_flatten = torch.cat(src_flatten, 2)
        mask_flatten = torch.cat(mask_flatten, 2)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 2)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m[:,0]) for m in masks], 1)  

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        # prepare input for decoder
        bs, nf,  _, c = memory.shape
        query_embed, tgt = torch.split(query_embed, c, dim=1)
        query_embed = query_embed.unsqueeze(0).unsqueeze(1).repeat(bs, nf, 1, 1)
        tgt = tgt.unsqueeze(0).unsqueeze(1).repeat(bs, nf, 1, 1)

        reference_points = self.reference_points(query_embed).sigmoid()
        init_reference_out = reference_points

        hs, inter_references, inter_samples = self.decoder(tgt, reference_points, memory, spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)       

        return hs, memory, init_reference_out, inter_references, inter_samples, None, valid_ratios

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.d_model = d_model
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, 'encode')
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        src2 = self.self_attn(self.with_pos_embed(src, pos), None, reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.forward_ffn(src)
        return src

class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output

class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, query_temporal=None):
        super().__init__()

        self.d_model = d_model
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, 'decode_vm')
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.query_temporal = query_temporal
        if self.query_temporal is not None:
            self.dropout_tem = nn.Dropout(dropout)
            if self.query_temporal == 'weight_sum':
                self.time_attention_weights = nn.Linear(d_model, 1)
            elif self.query_temporal == 'weight_sum_all':
                self.time_attention_weights = nn.Linear(d_model, d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    @staticmethod
    def with_pos_embed_multf(tensor, pos):
        return tensor if pos is None else tensor + pos.unsqueeze(1)

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        #assert len(tgt.shape) == 4
        if len(tgt.shape) == 3:
            q_tgt = k_tgt = self.with_pos_embed(tgt, query_pos)
            tgt2 = self.self_attn(q_tgt.transpose(0, 1), k_tgt.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
            tgt = tgt+ self.dropout2(tgt2)
            tgt = self.norm2(tgt)
            tgt2, sampling_locations, attention_weights = self.cross_attn(self.with_pos_embed(tgt, query_pos), None,
                                reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask)
        else:
            N, nf, num_q, C = tgt.shape
            tgt_list = []
            for i_f in range(nf):
                tgt_i =  tgt[:,i_f]
                query_pos_i = query_pos[:,i_f]
                q_tgt_i = k_tgt_i = self.with_pos_embed(tgt_i, query_pos_i)
                tgt2_i = self.self_attn(q_tgt_i.transpose(0, 1), k_tgt_i.transpose(0, 1), tgt_i.transpose(0, 1))[0].transpose(0, 1)
                tgt_i = tgt_i + self.dropout2(tgt2_i)
                tgt_i = self.norm2(tgt_i)
                tgt_list.append(tgt_i.unsqueeze(1))
            tgt = torch.cat(tgt_list,dim=1)
            tgt2, sampling_locations, attention_weights = self.cross_attn(self.with_pos_embed(tgt, query_pos), None,
                                reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask)
        
        if self.query_temporal == 'weight_sum' or self.query_temporal == 'weight_sum_all':
            time_weight = self.time_attention_weights(tgt2)
            time_weight = F.softmax(time_weight, 1)
            tgt_temporal = (tgt2*time_weight).sum(1).unsqueeze(1)

        if len(tgt.shape) == 3: 
            tgt = tgt.unsqueeze(1) + self.dropout1(tgt2)
        elif self.query_temporal == 'weight_sum' or self.query_temporal == 'weight_sum_all':
            tgt = tgt + self.dropout1(tgt2) + self.dropout_tem(tgt_temporal)
        else:
            tgt = tgt + self.dropout1(tgt2)

        tgt = self.norm1(tgt)
        
        tgt = self.forward_ffn(tgt)

        return tgt, sampling_locations, attention_weights

class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        
        output = tgt
        intermediate = []
        intermediate_reference_points = []
        intermediate_samples = []
        
        for lid, layer in enumerate(self.layers):
            
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None, None] 
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, :, None] * src_valid_ratios[:,None, None] 
            
            output, sampling_locations, attention_weights = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), None 

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_vm(args):
    return DeformableVMFormer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        query_temporal=args.query_temporal,
        num_frames=args.num_frames,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,)




