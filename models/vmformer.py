# ------------------------------------------------------------------------
# VMFormer model and criterion classes.
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

import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
                           
from .deformable_vm import build_deforamble_vm
from .loss import dice_loss, sigmoid_focal_loss, l1_loss, laplacian_loss, temporal_loss
import copy

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class VMFormer(nn.Module):
    def __init__(self, backbone, transformer, num_frames, num_queries, num_feature_levels, aux_loss=True, fpn_temporal=False):
        super().__init__()
        self.num_frames = num_frames
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.transformer.decoder.bbox_embed = None
        
        if self.backbone.modelname == 'mv3':
            self.mask_head = MaskHeadSmallConv(hidden_dim, hidden_dim, True, fpn_temporal)
        else:
            self.mask_head = MaskHeadSmallConv(hidden_dim, hidden_dim, False, fpn_temporal)

    def forward(self, samples: NestedTensor, gt_targets, criterion, train=False):
        if not isinstance(samples, NestedTensor):      
            samples = nested_tensor_from_tensor_list(samples, size_divisibility=32)     
        features, pos = self.backbone(samples)
        srcs = []
        masks = []
        poses = []
        spatial_shapes = []
        if self.backbone.modelname == 'mv3':
            largefeature = features[0].decompose()[0]
            features = features[1:]
            pos = pos[1:]
        else:
            largefeature = None
        for l, feat in enumerate(features):
            src, mask = feat.decompose() 
            src_proj_l = self.input_proj[l](src)    # src_proj_l: [nf*N, C, Hi, Wi]
            
            # src_proj_l -> [nf, N, C, Hi, Wi]
            n, c, h, w = src_proj_l.shape
            spatial_shapes.append((h, w))
            src_proj_l = src_proj_l.reshape(n//self.num_frames, self.num_frames, c, h, w)

            # mask -> [nf, N, Hi, Wi]
            mask = mask.reshape(n//self.num_frames, self.num_frames, h, w)
            
            # pos -> [nf, N, Hi, Wi]
            np, cp, hp, wp = pos[l].shape
            pos_l = pos[l].reshape(np//self.num_frames, self.num_frames, cp, hp, wp)
            
            srcs.append(src_proj_l)
            masks.append(mask)
            poses.append(pos_l)
            assert mask is not None
        query_embeds = None
        query_embeds = self.query_embed.weight

        hs, memory, init_reference, inter_references, inter_samples, enc_outputs_class, valid_ratios = self.transformer(srcs, masks, poses, query_embeds)

        valid_ratios = valid_ratios[:,0] 

        outputs = {}
        outputs_lvl_masks = []
        enc_lay_num = hs.shape[0]
        for lvl in range(enc_lay_num):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            # mask prediction
            lvl_masks = self.forward_mask_head_train(hs[lvl], memory, spatial_shapes, reference, largefeature)
            outputs_lvl_masks.append(lvl_masks)
        outputs_mask = outputs_lvl_masks
        outputs['pred_masks'] = outputs_mask[-1]
        if self.aux_loss:
            outputs['aux_outputs'] = self._set_aux_loss(outputs_mask)

        loss_dict = criterion(outputs, gt_targets, valid_ratios)

        return outputs, loss_dict

    def inference(self, samples: NestedTensor, orig_w, orig_h):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        srcs = []
        masks = []
        poses = []
        spatial_shapes = []

        if self.backbone.modelname == 'mv3':
            largefeature = features[0].decompose()[0]
            features = features[1:]
            pos = pos[1:]
        else:
            largefeature = None

        for l, feat in enumerate(features):
            src, mask = feat.decompose() 
            src_proj_l = self.input_proj[l](src)    # src_proj_l: [nf*N, C, Hi, Wi]
            
            # src_proj_l -> [nf, N, C, Hi, Wi]
            n, c, h, w = src_proj_l.shape
            spatial_shapes.append((h, w))
            src_proj_l = src_proj_l.reshape(n//self.num_frames, self.num_frames, c, h, w)
            
            # mask -> [nf, N, Hi, Wi]
            mask = mask.reshape(n//self.num_frames, self.num_frames, h, w)
            
            # pos -> [nf, N, Hi, Wi]
            np, cp, hp, wp = pos[l].shape
            pos_l = pos[l].reshape(np//self.num_frames, self.num_frames, cp, hp, wp)
            
            srcs.append(src_proj_l)
            masks.append(mask)
            poses.append(pos_l)
            assert mask is not None
        query_embeds = None
        query_embeds = self.query_embed.weight
       
        hs, memory, init_reference, inter_references, inter_samples, enc_outputs_class, valid_ratios = self.transformer(srcs, masks, poses, query_embeds)
        dec_lay_num = hs.shape[0]
        
        reference = inter_references[-1]
        # mask prediction
        lvl_masks = self.forward_mask_head_train(hs[-1], memory, spatial_shapes, reference, largefeature)

        return [lvl_mask[-1] for lvl_mask in lvl_masks]

    def forward_mask_head_train(self, outputs, feats, spatial_shapes, reference_points, largefeature):
        bs,n_f, _, c = feats.shape
        # nq = mask_head_params.shape[1]

        # encod_feat_l: num_layers x [bs, C, num_frames, hi, wi]
        encod_feat_l = []
        spatial_indx = 0

        ### bring feature pyramids back
        for feat_l in range(self.num_feature_levels):
            h, w = spatial_shapes[feat_l]
            mem_l = feats[:,:, spatial_indx: spatial_indx + h * w, :].reshape(bs, self.num_frames, h, w, c).permute(0,4,1,2,3)
            encod_feat_l.append(mem_l)
            spatial_indx += h * w
        ### decode features into frame-level spaces
        pred_masks = []
        tmp_feature = None
        
        if largefeature is not None:
            _, C, H, W = largefeature.shape
            largefeature = largefeature.reshape(bs, self.num_frames, C, H, W)

        for iframe in range(self.num_frames):
            encod_feat_f = []
            for lvl in range(self.num_feature_levels):
                encod_feat_f.append(encod_feat_l[lvl][:, :, iframe, :, :]) # [bs, C, hi, wi]
            if largefeature is not None:
                decod_feat_f, tmp_feature = self.mask_head(encod_feat_f, tmp_feature, largef=largefeature[:,iframe,:,:,:])
            else:
                decod_feat_f, tmp_feature = self.mask_head(encod_feat_f, largef=None)
            query_frame_embed = outputs[:,iframe,:,:]
            # query_frame [bs, q, C]
            mask_pyramid = []
            for decod_feat in decod_feat_f:
                mask_f = torch.einsum("bqc,bchw->bqhw", query_frame_embed, decod_feat)
                mask_pyramid.append(mask_f)
            pred_masks.append(mask_pyramid)  
        return pred_masks

    @torch.jit.unused
    def _set_aux_loss(self, outputs_mask):
        return [{'pred_masks': a} for a in outputs_mask[:-1]]

class SetCriterion(nn.Module):
    def __init__(self, weight_dict, losses, focal_alpha=0.25, mask_out_stride=4, num_frames=1):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.mask_out_stride = mask_out_stride
        self.num_frames = num_frames
        self.valid_ratios = None

    def loss_masks(self, outputs, targets):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        src_masks = outputs["pred_masks"]
        bs = len(targets)
        num_frames = len(src_masks)
        num_feature = len(src_masks[0])
        out_src_masks = []
        for i in range(num_feature):
            out_src_masks.append(torch.cat([src_mask[i] for src_mask in src_masks], dim=1))
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets], 
                                                             size_divisibility=32,
                                                             split=False).decompose()
        target_masks = target_masks.to(src_masks[0][0])
        im_h, im_w = target_masks.shape[-2:]
        losses = {}
        target_masks = target_masks.reshape(bs, num_frames, -1, target_masks.shape[-2], target_masks.shape[-1])
        for i, out_src_mask in enumerate(out_src_masks):
            out_src_mask = interpolate(out_src_mask, size=target_masks.shape[-2:], mode="bilinear", align_corners=False)
            out_src_mask  = out_src_mask.reshape(bs, num_frames, -1, out_src_mask.shape[-2], out_src_mask.shape[-1])
            #losses.update({
            #    "loss_l1"+f'_{i}': l1_loss(out_src_mask, target_masks),
            #    "loss_lap"f'_{i}': laplacian_loss(out_src_mask, target_masks),
            #    "loss_temporal"f'_{i}': temporal_loss(out_src_mask, target_masks)
            #})
            losses.update({
                "loss_mask"+f'_{i}': sigmoid_focal_loss(out_src_mask, target_masks),
                "loss_dice"f'_{i}': dice_loss(out_src_mask, target_masks),
                "loss_temporal"f'_{i}': temporal_loss(out_src_mask, target_masks)
            })
        return losses

    def forward(self, outputs, targets, valid_ratios):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        self.valid_ratios = valid_ratios

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.loss_masks(outputs, targets))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                for loss in self.losses:
                    l_dict = self.loss_masks(aux_outputs, targets)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses

class ConvTmp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.tmp_conv = torch.nn.Conv2d(dim, dim, 1, 1, 0)
        self.cur_conv = torch.nn.Conv2d(dim, dim, 1, 1, 0)
        self.ac = nn.Tanh()

    def forward(self, tmp_f, cur_f):
        tmp = self.ac(self.tmp_conv(tmp_f) + self.cur_conv(cur_f))
        return tmp, tmp

class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """
    def __init__(self, dim, context_dim, large_feature=False, fpn_temporal=False):
        super().__init__()
        self.lay2 = torch.nn.Conv2d(context_dim, context_dim, 3, padding=1)
        self.lay3 = torch.nn.Conv2d(context_dim, context_dim, 3, padding=1)
        self.lay4 = torch.nn.Conv2d(context_dim, context_dim, 3, padding=1)
        self.dim = dim
        self.context_dim = context_dim
        self.fpn_temporal = fpn_temporal
        
        if self.fpn_temporal:
            self.temporal4 = ConvTmp(context_dim//2)
            self.temporal3 = ConvTmp(context_dim//2)
            self.temporal2 = ConvTmp(context_dim//2)

        self.large_feature = large_feature
            
        if self.large_feature:
            self.proj = nn.Sequential(
                nn.Conv2d(16, context_dim, kernel_size=1),
                nn.GroupNorm(32, context_dim),
            )
            self.lay_up = torch.nn.Conv2d(context_dim, context_dim, 3, padding=1)

        for name, m in self.named_modules():
            if name == "conv_offset":
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
            else:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(m.weight, a=1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, tmp_feature, largef):
        out = []
        if self.fpn_temporal:
            current_tmp_feature = []

        fused_x = x[-1]
        
        if self.fpn_temporal:
            fused_x_a, fused_x_r = fused_x.split(self.context_dim // 2, dim=1)
            if tmp_feature is not None:
                tmp_f = tmp_feature[0]
                tmp_f_update, fused_x_r = self.temporal4(tmp_f, fused_x_r)
                fused_x = torch.cat([fused_x_a, fused_x_r], dim=1)
            else:
                tmp_f_update = fused_x_r
            current_tmp_feature.append(tmp_f_update)
        
        fused_x = self.lay4(fused_x)
        fused_x = F.relu(fused_x)

        fused_x = x[-2] + F.interpolate(fused_x, size=x[-2].shape[-2:], mode="bilinear", align_corners=False)
        
        if self.fpn_temporal:
            fused_x_a, fused_x_r = fused_x.split(self.context_dim // 2, dim=1)
            if tmp_feature is not None:
                tmp_f = tmp_feature[1]
                tmp_f_update, fused_x_r = self.temporal3(tmp_f, fused_x_r)
                fused_x = torch.cat([fused_x_a, fused_x_r], dim=1)
            else:
                tmp_f_update = fused_x_r
            current_tmp_feature.append(tmp_f_update)

        fused_x = self.lay3(fused_x)
        fused_x = F.relu(fused_x)

        fused_x = x[-3] + F.interpolate(fused_x, size=x[-3].shape[-2:], mode="bilinear", align_corners=False)
        
        if self.fpn_temporal:
            fused_x_a, fused_x_r = fused_x.split(self.context_dim // 2, dim=1)
            if tmp_feature is not None:
                tmp_f = tmp_feature[2]
                tmp_f_update, fused_x_r = self.temporal2(tmp_f, fused_x_r)
                fused_x = torch.cat([fused_x_a, fused_x_r], dim=1)
            else:
                tmp_f_update = fused_x_r
            current_tmp_feature.append(tmp_f_update)
            
        fused_x = self.lay2(fused_x)
        fused_x = F.relu(fused_x)

        if self.large_feature:
            fused_x = self.proj(largef) + F.interpolate(fused_x, size=largef.shape[-2:], mode="bilinear", align_corners=False)
            fused_x = self.lay_up(fused_x)
            fused_x = F.relu(fused_x)
        out.append(fused_x)

        if self.fpn_temporal: 
            return out, current_tmp_feature
        else:
            return out, None

def build_vm(args):    
    device = torch.device(args.device)

    if 'swin' in args.backbone:
        from .swin_transformer import build_swin_backbone
        backbone = build_swin_backbone(args) 
    else:
        backbone = build_backbone(args)

    transformer = build_deforamble_vm(args)

    model = VMFormer(
        backbone,
        transformer,
        num_frames=args.num_frames,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        fpn_temporal=args.fpn_temporal,
    )
    
    weight_dict = {'loss_mask': args.mask_loss_coef, 'loss_dice': args.dice_loss_coef, 'loss_l1': args.l1_loss_coef,
                    'loss_lap': args.lap_loss_coef, 'loss_temporal': args.temporal_loss_coef}
    
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['masks']

    criterion = SetCriterion(weight_dict, losses, 
                             mask_out_stride=args.mask_out_stride,
                             focal_alpha=args.focal_alpha,
                             num_frames = args.num_frames)
    criterion.to(device)
    
    postprocessors = []

    return model, criterion, postprocessors



