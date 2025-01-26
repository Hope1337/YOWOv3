import torch
from torch.nn.functional import cross_entropy, one_hot
import math
from utils.box import make_anchors
import cv2
import torch.nn.functional as F

class TAL:
    def __init__(self, model, config):
        super().__init__()
        if hasattr(model, 'module'):
            model = model.module

        device = next(model.parameters()).device  # get model device

        m = model.detection_head  # Head() module
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.device = device

        # task aligned assigner
        self.top_k          = config['LOSS']['TAL']['top_k']
        self.alpha          = config['LOSS']['TAL']['alpha']
        self.beta           = config['LOSS']['TAL']['beta']
        self.radius         = config['LOSS']['TAL']['radius']
        self.scale_cls_loss = config['LOSS']['TAL']['scale_cls_loss']
        self.scale_box_loss = config['LOSS']['TAL']['scale_box_loss']
        self.scale_dfl_loss = config['LOSS']['TAL']['scale_dfl_loss']
        self.soft_label     = config['LOSS']['TAL']['soft_label']
        self.eps            = 1e-9 

        self.bs = 1
        self.num_max_boxes = 0
        # DFL Loss params
        self.dfl_ch = m.dfl.ch
        self.project = torch.arange(self.dfl_ch, dtype=torch.float, device=device)
    #                                 img_idx + 4 corrdinate + nclass
    #                           [nbox, 1 + 4 + nclass]
    def __call__(self, outputs, targets):

        # [B, 4 * n_dfl_channel + num_classes, 14, 14]  (28, 14, 7)
        x = outputs[1] if isinstance(outputs, tuple) else outputs

        # [B, 4 * n_dfl_channel, 1029]
        output = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], 2)

        # [B, 4 * n_dfl_channel, 1029], [B, num_classes, 1029]
        pred_output, pred_scores = output.split((4 * self.dfl_ch, self.nc), 1)

        # [B, 1029, 4 * n_dfl_channel]
        pred_output = pred_output.permute(0, 2, 1).contiguous()

        # [B, 1029, num_classes]
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        nclass = pred_scores.shape[2]


        size = torch.tensor(x[0].shape[2:], dtype=pred_scores.dtype, device=self.device)
        size = size * self.stride[0]

        anchor_points, stride_tensor = make_anchors(x, self.stride, 0.5)

        # targets
        if targets.shape[0] == 0:
            gt = torch.zeros(pred_scores.shape[0], 0, 4 + nclass, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            gt = torch.zeros(pred_scores.shape[0], counts.max(), 4 +  nclass, device=self.device)
            for j in range(pred_scores.shape[0]):
                matches = i == j
                n = matches.sum()
                if n:
                    gt[j, :n] = targets[matches, 1:]
            #gt[..., 1:5] = wh2xy(gt[..., 1:5].mul_(size[[1, 0, 1, 0]]))
            gt[..., 0:4] = gt[..., 0:4].mul_(size[[1, 0, 1, 0]])

        gt_bboxes, gt_labels = gt.split((4, nclass), 2)  # cls, xyxy
        
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # boxes
        # [B, 1029, 4 * n_dfl_channel]
        b, a, c = pred_output.shape
        pred_bboxes = pred_output.view(b, a, 4, c // 4).softmax(3)

        # [B, 1029, 4] -> after decode
        pred_bboxes = pred_bboxes.matmul(self.project.type(pred_bboxes.dtype))

        a, b = torch.split(pred_bboxes, 2, -1)

        # [B, 1029, 4] 
        pred_bboxes = torch.cat((anchor_points - a, anchor_points + b), -1)

        # [B, 1029, num_classes] 
        scores = pred_scores.detach().sigmoid()

        # [B, 1029, 4] 
        bboxes = (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype)
        target_bboxes, target_scores, fg_mask = self.assign(scores, bboxes,
                                                            gt_labels, gt_bboxes, mask_gt,
                                                            anchor_points * stride_tensor, stride_tensor)
        
        mask = target_scores.gt(0)[fg_mask]

        target_bboxes /= stride_tensor
        target_scores_sum = target_scores.sum()
        #num_pos = fg_mask.sum()

        # cls loss
        loss_cls = self.bce(pred_scores, target_scores.to(pred_scores.dtype))
        loss_cls = loss_cls.sum() / target_scores_sum
        #loss_cls = loss_cls.sum() / num_pos

        # box loss
        loss_box = torch.zeros(1, device=self.device)
        loss_dfl = torch.zeros(1, device=self.device)
        if fg_mask.sum():
            # IoU loss
            weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
            loss_box = self.iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
            loss_box = ((1.0 - loss_box) * weight).sum() / target_scores_sum
            #loss_box = (1.0 - loss_box).sum() / num_pos
            # DFL loss
            a, b = torch.split(target_bboxes, 2, -1)
            target_lt_rb = torch.cat((anchor_points - a, b - anchor_points), -1)
            target_lt_rb = target_lt_rb.clamp(0, self.dfl_ch - 1.01)  # distance (left_top, right_bottom)
            loss_dfl = self.df_loss(pred_output[fg_mask].view(-1, self.dfl_ch), target_lt_rb[fg_mask])
            loss_dfl = (loss_dfl * weight).sum() / target_scores_sum
            #loss_dfl = loss_dfl.sum() / num_pos


        loss_cls *= self.scale_cls_loss
        loss_box *= self.scale_box_loss
        loss_dfl *= self.scale_dfl_loss

        #print("cls : {}, box : {}, dfl : {}".format(loss_cls.item(), loss_box.item(), loss_dfl.item()))
        return loss_cls + loss_box + loss_dfl  # loss(cls, box, dfl)

    @torch.no_grad()
    def assign(self, pred_scores, pred_bboxes, true_labels, true_bboxes, true_mask, anchors, stride_tensor):
        # print(pred_scores.shape) [B, 1029, nclass]
        # print(pred_bboxes.shape) [B, 1029, 4]
        # print(true_bboxes.shape) [B, nbox, 4]
        # print(true_labels.shape) #[B, nbox, nclass]
        # print(true_mask.shape) #[B, nbox, 1]
        # print(anchors.shape) #[1029, 2]
        # print(stride_anchor.shape) [1029, 1]
        # there are some fake box for shape purpose

        """
        Task-aligned One-stage Object Detection assigner
        """
        self.bs = pred_scores.size(0)
        self.num_max_boxes = true_bboxes.size(1)

        # need to be changed !
        ###################################################################################
        if self.num_max_boxes == 0:
            device = true_bboxes.device
            return (torch.full_like(pred_scores[..., 0], self.nc).to(device),
                    torch.zeros_like(pred_bboxes).to(device),
                    torch.zeros_like(pred_scores).to(device),
                    torch.zeros_like(pred_scores[..., 0]).to(device),
                    torch.zeros_like(pred_scores[..., 0]).to(device))
        ##################################################################################

        # [2, B, nbox]
        #i = torch.zeros([2, self.bs, self.num_max_boxes], dtype=torch.long)
        #i[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.num_max_boxes)
        #i[1] = true_labels.long()

        # [B, nbox, 1029, 1]
        overlaps = self.iou(true_bboxes.unsqueeze(2), pred_bboxes.unsqueeze(1))
        # [B, nbox, 1029]
        overlaps = overlaps.squeeze(3).clamp(0)

        # [B, nbox, 1029] [B, 1029, nclass] [B, nbox, nclass]
        #align_metric = pred_scores[i[0], :, i[1]].pow(self.alpha) * overlaps.pow(self.beta)

        # [B, 1029, nbox, nclass]
        pred_scores_loss = pred_scores.unsqueeze(-2).repeat([1, 1, overlaps.shape[1], 1])

        # [B, 1029, nbox, nclass]
        true_scores      = true_labels.unsqueeze(1).repeat([1, pred_scores_loss.shape[1], 1, 1])

        # [B, 1029, nbox]
        scaler = true_scores.sum(-1)
        scaler = scaler.clamp(min=1)

        # [B, nbox, 1029]
        scores_loss = ((true_scores * pred_scores_loss).sum(-1) / scaler).permute(0, 2, 1).contiguous()
        
        align_metric = scores_loss.pow(self.alpha) * overlaps.pow(self.beta)
        
        bs, n_boxes, _ = true_bboxes.shape

        # [B * nbox, 1, 2]
        lt, rb = true_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom

        # [B * nbox, 1029, 4], offset for box regression
        bbox_deltas = torch.cat((anchors[None] - lt, rb - anchors[None]), dim=2)

        # [B, nbox, 1029]
        mask_in_gts = bbox_deltas.view(bs, n_boxes, anchors.shape[0], -1).amin(3).gt_(1e-9)
        
        # [1029]
        radius = (stride_tensor * self.radius).squeeze(-1)

        # # print(anchors.shape) #[1029, 2]
        # [B * nbox, 1, 2]
        gt_center = (rb - lt) / 2.

        # [B, nbox, 1029]
        center_distance = torch.sqrt((gt_center - anchors[None]).pow(2).sum(-1)).view(bs, n_boxes, -1)

        mask_in_radius = (center_distance - radius).gt(self.eps)

        # [B, nbox, 1029]
        metrics = align_metric * mask_in_gts * mask_in_radius

        # [B, nbox, top_k]
        top_k_mask = true_mask.repeat([1, 1, self.top_k]).bool()
        
        num_anchors = metrics.shape[-1]

        # [B, nbox, top_k]
        top_k_metrics, top_k_indices = torch.topk(metrics, self.top_k, dim=-1, largest=True)
    
        if top_k_mask is None:
            top_k_mask = (top_k_metrics.max(-1, keepdim=True) > self.eps).tile([1, 1, self.top_k])
        
        # [B, nbox, top_k]
        top_k_indices = torch.where(top_k_mask, top_k_indices, 0)
        
        # [B, nbox, 1029]
        is_in_top_k = one_hot(top_k_indices, num_anchors).sum(-2)

        # filter invalid boxes
        is_in_top_k = torch.where(is_in_top_k > 1, 0, is_in_top_k)
        mask_top_k = is_in_top_k.to(metrics.dtype)
        # merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_top_k * mask_in_gts * mask_in_radius * true_mask

        # [B, 1029]
        fg_mask = mask_pos.sum(-2)

        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes

            # [B, nbox, 1029]
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, self.num_max_boxes, 1])

            # [B, 1029]
            max_overlaps_idx = overlaps.argmax(1)

            # [B, 1029, nbox]
            is_max_overlaps = one_hot(max_overlaps_idx, self.num_max_boxes)

            # [B, nbox, 1029]
            is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)
            
            # [B, nbox, 1029] 
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)

            # [B, 1029]
            fg_mask = mask_pos.sum(-2)
            
        # find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)

        # assigned target labels, (b, 1)
        batch_index = torch.arange(end=self.bs,
                                   dtype=torch.int64,
                                   device=true_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_index * self.num_max_boxes

        # [B, 1029, nclass]
        target_labels = true_labels.long().view(-1, true_labels.shape[-1])[target_gt_idx]

        # assigned target boxes
        target_bboxes = true_bboxes.view(-1, 4)[target_gt_idx]

        # assigned target scores
        target_labels.clamp(0)
        #target_scores = one_hot(target_labels, self.nc)
        target_scores = target_labels
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        if self.soft_label:
            # normalize
            align_metric *= mask_pos
            pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)
            pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)
            norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2)
            norm_align_metric = norm_align_metric.unsqueeze(-1)
            target_scores = target_scores * norm_align_metric 
            pos_overlaps = (overlaps * mask_pos).sum(-2).unsqueeze(-1)
            target_scores = target_scores * pos_overlaps

        return target_bboxes, target_scores, fg_mask.bool()

    @staticmethod
    def df_loss(pred_dist, target):
        # Return sum of left and right DFL losses
        # Distribution Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        l_loss = cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape)
        r_loss = cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape)
        return (l_loss * wl + r_loss * wr).mean(-1, keepdim=True)

    @staticmethod
    def iou(box1, box2, eps=1e-7):
        # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

        # Intersection area
        area1 = b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)
        area2 = b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
        intersection = area1.clamp(0) * area2.clamp(0)

        # Union Area
        union = w1 * h1 + w2 * h2 - intersection + eps

        # IoU
        iou = intersection / union
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        # Complete IoU https://arxiv.org/abs/1911.08287v1
        c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
        # center dist ** 2
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
        # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
        v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - (rho2 / c2 + v * alpha)  # CIoU
    
class SimOTA:
    def __init__(self, model, config):
        super().__init__()
        if hasattr(model, 'module'):
            model = model.module

        device = next(model.parameters()).device  # get model device

        m = model.detection_head  # Head() module
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.device = device

        # task aligned assigner
        self.top_k          = config['LOSS']['SIMOTA']['top_k']
        self.radius         = config['LOSS']['SIMOTA']['radius']
        self.mode           = config['LOSS']['SIMOTA']['mode']
        self.scale_cls_loss = config['LOSS']['SIMOTA']['scale_cls_loss']
        self.scale_box_loss = config['LOSS']['SIMOTA']['scale_box_loss']
        self.scale_dfl_loss = config['LOSS']['SIMOTA']['scale_dfl_loss']
        self.gamma          = config['LOSS']['SIMOTA']['gamma']
        self.dynamic_k      = config['LOSS']['SIMOTA']['dynamic_k']
        self.dynamic_top_k  = config['LOSS']['SIMOTA']['dynamic_top_k']
        self.soft_label     = config['LOSS']['SIMOTA']['soft_label']
        self.eps            = 1e-9

        if self.mode == 'unbalance':
            ratio_dict = config['class_ratio']
            self.class_weights = torch.zeros(len(list(ratio_dict.keys())))
            for x in ratio_dict.keys():
                self.class_weights[x] = 1.0 - ratio_dict[x]

            self.class_weights = self.class_weights.to('cuda')

        self.bs = 1
        self.num_max_boxes = 0
        # DFL Loss params
        self.dfl_ch = m.dfl.ch
        self.project = torch.arange(self.dfl_ch, dtype=torch.float, device=device)
    #                                 img_idx + 4 corrdinate + nclass
    #                           [nbox, 1 + 4 + nclass]
    def __call__(self, outputs, targets):

        # [B, 4 * n_dfl_channel + num_classes, 14, 14]  (28, 14, 7)
        x = outputs[1] if isinstance(outputs, tuple) else outputs

        # [B, 4 * n_dfl_channel, 1029]
        output = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], 2)

        # [B, 4 * n_dfl_channel, 1029], [B, num_classes, 1029]
        pred_output, pred_scores = output.split((4 * self.dfl_ch, self.nc), 1)

        # [B, 1029, 4 * n_dfl_channel]
        pred_output = pred_output.permute(0, 2, 1).contiguous()

        # [B, 1029, num_classes]
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        nclass = pred_scores.shape[2]


        size = torch.tensor(x[0].shape[2:], dtype=pred_scores.dtype, device=self.device)
        size = size * self.stride[0]

        anchor_points, stride_tensor = make_anchors(x, self.stride, 0.5)

        # targets
        if targets.shape[0] == 0:
            gt = torch.zeros(pred_scores.shape[0], 0, 4 + nclass, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            gt = torch.zeros(pred_scores.shape[0], counts.max(), 4 +  nclass, device=self.device)
            for j in range(pred_scores.shape[0]):
                matches = i == j
                n = matches.sum()
                if n:
                    gt[j, :n] = targets[matches, 1:]
            #gt[..., 1:5] = wh2xy(gt[..., 1:5].mul_(size[[1, 0, 1, 0]]))
            gt[..., 0:4] = gt[..., 0:4].mul_(size[[1, 0, 1, 0]])

        gt_bboxes, gt_labels = gt.split((4, nclass), 2)  # cls, xyxy
        
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # boxes
        # [B, 1029, 4 * n_dfl_channel]
        b, a, c = pred_output.shape
        pred_bboxes = pred_output.view(b, a, 4, c // 4).softmax(3)

        # [B, 1029, 4] -> after decode
        pred_bboxes = pred_bboxes.matmul(self.project.type(pred_bboxes.dtype))

        a, b = torch.split(pred_bboxes, 2, -1)

        # [B, 1029, 4] 
        pred_bboxes = torch.cat((anchor_points - a, anchor_points + b), -1)

        # [B, 1029, num_classes] 
        scores = pred_scores.detach().sigmoid()

        # [B, 1029, 4] 
        bboxes = (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype)
        target_bboxes, target_scores, fg_mask = self.assign(scores, bboxes,
                                                            gt_labels, gt_bboxes, mask_gt,
                                                            anchor_points * stride_tensor, stride_tensor)
        
        mask = target_scores.gt(0)[fg_mask]

        target_bboxes /= stride_tensor 
        #target_scores_sum = target_scores.sum()
        num_pos = fg_mask.sum()
        
        # cls loss
        pos_scores = pred_scores[fg_mask][mask].sigmoid()
        neg_scores = torch.cat((pred_scores[fg_mask][~mask], pred_scores[~fg_mask].view(-1)), dim=0).sigmoid()

        if self.mode == 'unbalance':
            pos_weight  = self.class_weights.unsqueeze(0).repeat([mask.shape[0], 1])[mask]

            neg_weight1 = self.class_weights.unsqueeze(0).repeat([mask.shape[0], 1])[~mask]  
            neg_weight2 = self.class_weights.unsqueeze(0).repeat([pred_scores[~fg_mask].shape[0], 1]).view(-1)
            neg_weight  = torch.cat((neg_weight1, neg_weight2))

            pos_weight = torch.exp(pos_weight)
            neg_weight = torch.exp(1.0 - neg_weight)

        elif self.mode == 'balance':
            pos_weight = 1.0
            neg_weight = 1.0

        gamma = self.gamma
            
        tg       = target_scores[fg_mask][mask]
        pos1     = - pos_weight * torch.clamp(torch.abs(tg - pos_scores), min=self.eps) ** gamma
        pos2     = tg*torch.log(pos_scores + self.eps) + (1. - tg)*torch.log(1. - pos_scores + self.eps)
        pos_loss = pos1*pos2
        neg_loss = - neg_weight * torch.clamp(neg_scores, min=self.eps) ** gamma * torch.log(1. - neg_scores + self.eps)
        loss_cls = (pos_loss.sum() + neg_loss.sum()) / num_pos
        #loss_cls = self.bce(pred_scores, target_scores.to(pred_scores.dtype))
        #loss_cls = loss_cls.sum() / target_scores_sum

        # box loss
        loss_box = torch.zeros(1, device=self.device)
        loss_dfl = torch.zeros(1, device=self.device)
        if fg_mask.sum():
            # IoU loss
            #weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
            loss_box = self.iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
            #loss_box = ((1.0 - loss_box) * weight).sum() / target_scores_sum
            loss_box = (1.0 - loss_box).sum() / num_pos
            # DFL loss
            a, b = torch.split(target_bboxes, 2, -1)
            target_lt_rb = torch.cat((anchor_points - a, b - anchor_points), -1)
            target_lt_rb = target_lt_rb.clamp(0, self.dfl_ch - 1.01)  # distance (left_top, right_bottom)
            loss_dfl = self.df_loss(pred_output[fg_mask].view(-1, self.dfl_ch), target_lt_rb[fg_mask])
            #loss_dfl = (loss_dfl * weight).sum() / target_scores_sum
            loss_dfl = loss_dfl.sum() / num_pos


        loss_cls *= self.scale_cls_loss
        loss_box *= self.scale_box_loss
        loss_dfl *= self.scale_dfl_loss

        #print(loss_cls)
        #print(loss_box)
        #print(loss_dfl)
        #print()

        #print("cls : {}, box : {}, dfl : {}".format(loss_cls.item(), loss_box.item(), loss_dfl.item()))
        return loss_cls + loss_box + loss_dfl  # loss(cls, box, dfl)

    @torch.no_grad()
    def assign(self, pred_scores, pred_bboxes, true_labels, true_bboxes, true_mask, anchors, stride_tensor):
        # print(pred_scores.shape) [B, 1029, nclass]
        # print(pred_bboxes.shape) [B, 1029, 4]
        # print(true_bboxes.shape) [B, nbox, 4]
        # print(true_labels.shape) #[B, nbox, nclass]
        # print(true_mask.shape) #[B, nbox, 1]
        # print(anchors.shape) #[1029, 2]
        # print(stride_anchor.shape) [1029, 1]
        # there are some fake box for shape purpose

        """
        Task-aligned One-stage Object Detection assigner
        """
        self.bs = pred_scores.size(0)
        self.num_max_boxes = true_bboxes.size(1)

        # need to be changed !
        ###################################################################################
        if self.num_max_boxes == 0:
            device = true_bboxes.device
            return (torch.full_like(pred_scores[..., 0], self.nc).to(device),
                    torch.zeros_like(pred_bboxes).to(device),
                    torch.zeros_like(pred_scores).to(device),
                    torch.zeros_like(pred_scores[..., 0]).to(device),
                    torch.zeros_like(pred_scores[..., 0]).to(device))
        ##################################################################################

        # [2, B, nbox]
        #i = torch.zeros([2, self.bs, self.num_max_boxes], dtype=torch.long)
        #i[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.num_max_boxes)
        #i[1] = true_labels.long()

        # [B, nbox, 1029, 1]
        overlaps = self.iou(true_bboxes.unsqueeze(2), pred_bboxes.unsqueeze(1))
        # [B, nbox, 1029]
        overlaps = overlaps.squeeze(3).clamp(0)

        # [B, nbox, 1029] [B, 1029, nclass] [B, nbox, nclass]
        #align_metric = pred_scores[i[0], :, i[1]].pow(self.alpha) * overlaps.pow(self.beta)

        # [B, 1029, nbox, nclass]
        pred_scores_loss = pred_scores.unsqueeze(-2).repeat([1, 1, overlaps.shape[1], 1])*(overlaps.permute(0, 2, 1).contiguous().unsqueeze(-1))   

        # [B, 1029, nbox, nclass]
        true_scores      = true_labels.unsqueeze(1).repeat([1, pred_scores_loss.shape[1], 1, 1])

        # [B, 1029, nbox]
        #scaler = true_scores.sum(-1)
        #scaler = scaler.clamp(min=1)

        # [B, nbox, 1029]
        # print((true_scores*pred_scores_loss).sum(-1))

        # scores_loss = ((true_scores * pred_scores_loss).sum(-1)).permute(0, 2, 1).contiguous()
        scores_loss = ((F.binary_cross_entropy(pred_scores_loss, true_scores, reduction='none')).sum(-1)).permute(0, 2, 1).contiguous()

        #class_weights = self.class_weights.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        #pos_loss = - true_scores * torch.exp(class_weights) * torch.clamp(1. - pred_scores_loss, self.eps) ** self.gamma * torch.log(pred_scores_loss + self.eps)
        #neg_loss = - (1. - true_scores) * torch.exp(1. - class_weights) * torch.clamp(pred_scores_loss, self.eps) ** self.gamma * torch.log(1. - pred_scores_loss  + self.eps)
        #scores_loss = (pos_loss + neg_loss).sum(-1).permute(0, 2, 1).contiguous()

        iou_loss = -torch.log(overlaps + self.eps)
        
        align_metric = scores_loss + 3 * iou_loss
        #print(align_metric.min())
        #print(align_metric.max())
        #import sys
        #sys.exit()
        
        bs, n_boxes, _ = true_bboxes.shape

        # [B * nbox, 1, 2]
        lt, rb = true_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom

        # [B * nbox, 1029, 4], offset for box regression
        bbox_deltas = torch.cat((anchors[None] - lt, rb - anchors[None]), dim=2)

        # [B, nbox, 1029]
        mask_in_gts = bbox_deltas.view(bs, n_boxes, anchors.shape[0], -1).amin(3).gt_(1e-9)
        
        # [1029]
        radius = (stride_tensor * self.radius).squeeze(-1)

        # # print(anchors.shape) #[1029, 2]
        # [B * nbox, 1, 2]
        gt_center = (rb - lt) / 2.

        # [B, nbox, 1029]
        center_distance = torch.sqrt((gt_center - anchors[None]).pow(2).sum(-1)).view(bs, n_boxes, -1)

        mask_in_radius = (center_distance - radius).gt_(self.eps)

        # [B, nbox, 1029]
        metrics = align_metric + ((1 - mask_in_gts) + (1 - mask_in_radius) + (1 - true_mask))*1000000

        num_anchors = metrics.shape[-1]

        if self.dynamic_k == True:
            # [B, nbox, top_k]
            top_k_sum, _ = torch.topk(overlaps, self.dynamic_top_k, dim=-1, largest=True)

            # [B, nbox]
            top_k_sum    = torch.clamp(top_k_sum.sum(-1), min=1).int()
        
            top_k        = top_k_sum.max().item()

            # [B, nbox, top_k]
            top_k_metrics, top_k_indices = torch.topk(metrics, top_k, dim=-1, largest=False)
        else :
            # [B, nbox, top_k]
            top_k_metrics, top_k_indices = torch.topk(metrics, self.top_k, dim=-1, largest=False)
            top_k                        = self.top_k

        # [B, nbox, top_k]
        top_k_mask = true_mask.repeat([1, 1, top_k]).bool()
    
        if top_k_mask is None:
            top_k_mask = (top_k_metrics.min(-1, keepdim=True) > self.eps).tile([1, 1, top_k])
        
        # [B, nbox, top_k]
        top_k_indices = torch.where(top_k_mask, top_k_indices, 0)
        
        # [B, nbox, 1029]
        is_in_top_k = one_hot(top_k_indices, num_anchors).sum(-2)

        if self.dynamic_k == True:
            B, nbox, _ = is_in_top_k.shape
            for i in range(B):
                for j in range(nbox):
                    for x in top_k_indices[i][j][top_k_sum[i][j]:]:
                        is_in_top_k[i][j][x] = 0

        # filter invalid boxes
        is_in_top_k = torch.where(is_in_top_k > 1, 0, is_in_top_k)
        mask_top_k = is_in_top_k.to(metrics.dtype)
        # merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_top_k * mask_in_gts * mask_in_radius * true_mask

        # [B, 1029]
        fg_mask = mask_pos.sum(-2)

        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes

            # [B, nbox, 1029]
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, self.num_max_boxes, 1])

            # [B, 1029]
            #max_overlaps_idx = overlaps.argmax(1)
            min_metrics_idx  = metrics.argmin(1)

            # [B, 1029, nbox]
            #is_max_overlaps = one_hot(max_overlaps_idx, self.num_max_boxes)
            is_min_metrics  = one_hot(min_metrics_idx, self.num_max_boxes)

            # [B, nbox, 1029]
            #is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)
            is_min_metrics  = is_min_metrics.permute(0, 2, 1).to(metrics.dtype)
            
            # [B, nbox, 1029] 
            #mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
            mask_pos = torch.where(mask_multi_gts, is_min_metrics, mask_pos)

            # [B, 1029]
            fg_mask = mask_pos.sum(-2)
            
        # find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)

        # assigned target labels, (b, 1)
        batch_index = torch.arange(end=self.bs,
                                   dtype=torch.int64,
                                   device=true_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_index * self.num_max_boxes

        # [B, 1029, nclass]
        target_labels = true_labels.long().view(-1, true_labels.shape[-1])[target_gt_idx]

        # assigned target boxes
        target_bboxes = true_bboxes.view(-1, 4)[target_gt_idx]

        # assigned target scores
        target_labels.clamp(0)
        #target_scores = one_hot(target_labels, self.nc)
        target_scores = target_labels
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        if self.soft_label:
            pos_overlaps = (overlaps * mask_pos).sum(-2).unsqueeze(-1)
            target_scores = target_scores * pos_overlaps

        return target_bboxes, target_scores, fg_mask.bool()

    @staticmethod
    def df_loss(pred_dist, target):
        # Return sum of left and right DFL losses
        # Distribution Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        l_loss = cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape)
        r_loss = cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape)
        return (l_loss * wl + r_loss * wr).mean(-1, keepdim=True)

    @staticmethod
    def iou(box1, box2, eps=1e-7):
        # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

        # Intersection area
        area1 = b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)
        area2 = b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
        intersection = area1.clamp(0) * area2.clamp(0)

        # Union Area
        union = w1 * h1 + w2 * h2 - intersection + eps

        # IoU
        iou = intersection / union
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        # Complete IoU https://arxiv.org/abs/1911.08287v1
        c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
        # center dist ** 2
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
        # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
        v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - (rho2 / c2 + v * alpha)  # CIoU
    
class Normal:
    def __init__(self, model, config):
        super().__init__()
        if hasattr(model, 'module'):
            model = model.module

        device = next(model.parameters()).device  # get model device

        m = model.detection_head  # Head() module
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.device = device

        # task aligned assigner
        self.top_k          = config['LOSS']['NORMAL']['top_k']
        self.radius         = config['LOSS']['NORMAL']['radius']
        self.scale_cls_loss = config['LOSS']['NORMAL']['scale_cls_loss']
        self.scale_box_loss = config['LOSS']['NORMAL']['scale_box_loss']
        self.scale_dfl_loss = config['LOSS']['NORMAL']['scale_dfl_loss']
        self.dynamic_k      = config['LOSS']['NORMAL']['dynamic_k']
        self.dynamic_top_k  = config['LOSS']['NORMAL']['dynamic_top_k']
        self.eps            = 1e-9

        self.bs = 1
        self.num_max_boxes = 0
        # DFL Loss params
        self.dfl_ch = m.dfl.ch
        self.project = torch.arange(self.dfl_ch, dtype=torch.float, device=device)
    #                                 img_idx + 4 corrdinate + nclass
    #                           [nbox, 1 + 4 + nclass]
    def __call__(self, outputs, targets):

        # [B, 4 * n_dfl_channel + num_classes, 14, 14]  (28, 14, 7)
        x = outputs[1] if isinstance(outputs, tuple) else outputs

        # [B, 4 * n_dfl_channel, 1029]
        output = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], 2)

        # [B, 4 * n_dfl_channel, 1029], [B, num_classes, 1029]
        pred_output, pred_scores = output.split((4 * self.dfl_ch, self.nc), 1)

        # [B, 1029, 4 * n_dfl_channel]
        pred_output = pred_output.permute(0, 2, 1).contiguous()

        # [B, 1029, num_classes]
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        nclass = pred_scores.shape[2]


        size = torch.tensor(x[0].shape[2:], dtype=pred_scores.dtype, device=self.device)
        size = size * self.stride[0]

        anchor_points, stride_tensor = make_anchors(x, self.stride, 0.5)

        # targets
        if targets.shape[0] == 0:
            gt = torch.zeros(pred_scores.shape[0], 0, 4 + nclass, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            gt = torch.zeros(pred_scores.shape[0], counts.max(), 4 +  nclass, device=self.device)
            for j in range(pred_scores.shape[0]):
                matches = i == j
                n = matches.sum()
                if n:
                    gt[j, :n] = targets[matches, 1:]
            #gt[..., 1:5] = wh2xy(gt[..., 1:5].mul_(size[[1, 0, 1, 0]]))
            gt[..., 0:4] = gt[..., 0:4].mul_(size[[1, 0, 1, 0]])

        gt_bboxes, gt_labels = gt.split((4, nclass), 2)  # cls, xyxy
        
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # boxes
        # [B, 1029, 4 * n_dfl_channel]
        b, a, c = pred_output.shape
        pred_bboxes = pred_output.view(b, a, 4, c // 4).softmax(3)

        # [B, 1029, 4] -> after decode
        pred_bboxes = pred_bboxes.matmul(self.project.type(pred_bboxes.dtype))

        a, b = torch.split(pred_bboxes, 2, -1)

        # [B, 1029, 4] 
        pred_bboxes = torch.cat((anchor_points - a, anchor_points + b), -1)

        # [B, 1029, num_classes] 
        scores = pred_scores.detach().sigmoid()

        # [B, 1029, 4] 
        bboxes = (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype)
        target_bboxes, target_scores, fg_mask = self.assign(scores, bboxes,
                                                            gt_labels, gt_bboxes, mask_gt,
                                                            anchor_points * stride_tensor, stride_tensor)
        
        mask = target_scores.gt(0)[fg_mask]

        target_bboxes /= stride_tensor 
        #target_scores_sum = target_scores.sum()
        num_pos = fg_mask.sum()
            
        loss_cls = self.bce(pred_scores, target_scores.to(pred_scores.dtype))
        loss_cls = loss_cls.sum() / num_pos

        # box loss
        loss_box = torch.zeros(1, device=self.device)
        loss_dfl = torch.zeros(1, device=self.device)
        if fg_mask.sum():
            # IoU loss
            #weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
            loss_box = self.iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
            #loss_box = ((1.0 - loss_box) * weight).sum() / target_scores_sum
            loss_box = (1.0 - loss_box).sum() / num_pos
            # DFL loss
            a, b = torch.split(target_bboxes, 2, -1)
            target_lt_rb = torch.cat((anchor_points - a, b - anchor_points), -1)
            target_lt_rb = target_lt_rb.clamp(0, self.dfl_ch - 1.01)  # distance (left_top, right_bottom)
            loss_dfl = self.df_loss(pred_output[fg_mask].view(-1, self.dfl_ch), target_lt_rb[fg_mask])
            #loss_dfl = (loss_dfl * weight).sum() / target_scores_sum
            loss_dfl = loss_dfl.sum() / num_pos


        loss_cls *= self.scale_cls_loss
        loss_box *= self.scale_box_loss
        loss_dfl *= self.scale_dfl_loss

        #print(loss_cls)
        #print(loss_box)
        #print(loss_dfl)
        #print()

        #print("cls : {}, box : {}, dfl : {}".format(loss_cls.item(), loss_box.item(), loss_dfl.item()))
        return loss_cls + loss_box + loss_dfl  # loss(cls, box, dfl)

    @torch.no_grad()
    def assign(self, pred_scores, pred_bboxes, true_labels, true_bboxes, true_mask, anchors, stride_tensor):
        # print(pred_scores.shape) [B, 1029, nclass]
        # print(pred_bboxes.shape) [B, 1029, 4]
        # print(true_bboxes.shape) [B, nbox, 4]
        # print(true_labels.shape) #[B, nbox, nclass]
        # print(true_mask.shape) #[B, nbox, 1]
        # print(anchors.shape) #[1029, 2]
        # print(stride_anchor.shape) [1029, 1]
        # there are some fake box for shape purpose

        """
        Task-aligned One-stage Object Detection assigner
        """
        self.bs = pred_scores.size(0)
        self.num_max_boxes = true_bboxes.size(1)

        # need to be changed !
        ###################################################################################
        if self.num_max_boxes == 0:
            device = true_bboxes.device
            return (torch.full_like(pred_scores[..., 0], self.nc).to(device),
                    torch.zeros_like(pred_bboxes).to(device),
                    torch.zeros_like(pred_scores).to(device),
                    torch.zeros_like(pred_scores[..., 0]).to(device),
                    torch.zeros_like(pred_scores[..., 0]).to(device))
        ##################################################################################

        # [2, B, nbox]
        #i = torch.zeros([2, self.bs, self.num_max_boxes], dtype=torch.long)
        #i[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.num_max_boxes)
        #i[1] = true_labels.long()

        # [B, nbox, 1029, 1]
        overlaps = self.iou(true_bboxes.unsqueeze(2), pred_bboxes.unsqueeze(1))
        # [B, nbox, 1029]
        overlaps = overlaps.squeeze(3).clamp(0)

        # [B, nbox, 1029] [B, 1029, nclass] [B, nbox, nclass]
        #align_metric = pred_scores[i[0], :, i[1]].pow(self.alpha) * overlaps.pow(self.beta)

        # [B, 1029, nbox, nclass]
        pred_scores_loss = pred_scores.unsqueeze(-2).repeat([1, 1, overlaps.shape[1], 1])*(overlaps.permute(0, 2, 1).contiguous().unsqueeze(-1))   

        # [B, 1029, nbox, nclass]
        true_scores      = true_labels.unsqueeze(1).repeat([1, pred_scores_loss.shape[1], 1, 1])

        # [B, 1029, nbox]
        #scaler = true_scores.sum(-1)
        #scaler = scaler.clamp(min=1)

        # [B, nbox, 1029]
        # print((true_scores*pred_scores_loss).sum(-1))

        # scores_loss = ((true_scores * pred_scores_loss).sum(-1)).permute(0, 2, 1).contiguous()
        scores_loss = ((F.binary_cross_entropy(pred_scores_loss, true_scores, reduction='none')).sum(-1)).permute(0, 2, 1).contiguous()

        #class_weights = self.class_weights.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        #pos_loss = - true_scores * torch.exp(class_weights) * torch.clamp(1. - pred_scores_loss, self.eps) ** self.gamma * torch.log(pred_scores_loss + self.eps)
        #neg_loss = - (1. - true_scores) * torch.exp(1. - class_weights) * torch.clamp(pred_scores_loss, self.eps) ** self.gamma * torch.log(1. - pred_scores_loss  + self.eps)
        #scores_loss = (pos_loss + neg_loss).sum(-1).permute(0, 2, 1).contiguous()

        iou_loss = -torch.log(overlaps + self.eps)
        
        align_metric = scores_loss + 3 * iou_loss
        #print(align_metric.min())
        #print(align_metric.max())
        #import sys
        #sys.exit()
        
        bs, n_boxes, _ = true_bboxes.shape

        # [B * nbox, 1, 2]
        lt, rb = true_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom

        # [B * nbox, 1029, 4], offset for box regression
        bbox_deltas = torch.cat((anchors[None] - lt, rb - anchors[None]), dim=2)

        # [B, nbox, 1029]
        mask_in_gts = bbox_deltas.view(bs, n_boxes, anchors.shape[0], -1).amin(3).gt_(1e-9)
        
        # [1029]
        radius = (stride_tensor * self.radius).squeeze(-1)

        # # print(anchors.shape) #[1029, 2]
        # [B * nbox, 1, 2]
        gt_center = (rb - lt) / 2.

        # [B, nbox, 1029]
        center_distance = torch.sqrt((gt_center - anchors[None]).pow(2).sum(-1)).view(bs, n_boxes, -1)

        mask_in_radius = (center_distance - radius).gt_(self.eps)

        # [B, nbox, 1029]
        metrics = align_metric + ((1 - mask_in_gts) + (1 - mask_in_radius) + (1 - true_mask))*1000000

        num_anchors = metrics.shape[-1]

        if self.dynamic_k == True:
            # [B, nbox, top_k]
            top_k_sum, _ = torch.topk(overlaps, self.dynamic_top_k, dim=-1, largest=True)

            # [B, nbox]
            top_k_sum    = torch.clamp(top_k_sum.sum(-1), min=1).int()
        
            top_k        = top_k_sum.max().item()

            # [B, nbox, top_k]
            top_k_metrics, top_k_indices = torch.topk(metrics, top_k, dim=-1, largest=False)
        else :
            # [B, nbox, top_k]
            top_k_metrics, top_k_indices = torch.topk(metrics, self.top_k, dim=-1, largest=False)
            top_k                        = self.top_k

        # [B, nbox, top_k]
        top_k_mask = true_mask.repeat([1, 1, top_k]).bool()
    
        if top_k_mask is None:
            top_k_mask = (top_k_metrics.min(-1, keepdim=True) > self.eps).tile([1, 1, top_k])
        
        # [B, nbox, top_k]
        top_k_indices = torch.where(top_k_mask, top_k_indices, 0)
        
        # [B, nbox, 1029]
        is_in_top_k = one_hot(top_k_indices, num_anchors).sum(-2)

        if self.dynamic_k == True:
            B, nbox, _ = is_in_top_k.shape
            for i in range(B):
                for j in range(nbox):
                    for x in top_k_indices[i][j][top_k_sum[i][j]:]:
                        is_in_top_k[i][j][x] = 0

        # filter invalid boxes
        is_in_top_k = torch.where(is_in_top_k > 1, 0, is_in_top_k)
        mask_top_k = is_in_top_k.to(metrics.dtype)
        # merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_top_k * mask_in_gts * mask_in_radius * true_mask

        # [B, 1029]
        fg_mask = mask_pos.sum(-2)

        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes

            # [B, nbox, 1029]
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, self.num_max_boxes, 1])

            # [B, 1029]
            #max_overlaps_idx = overlaps.argmax(1)
            min_metrics_idx  = metrics.argmin(1)

            # [B, 1029, nbox]
            #is_max_overlaps = one_hot(max_overlaps_idx, self.num_max_boxes)
            is_min_metrics  = one_hot(min_metrics_idx, self.num_max_boxes)

            # [B, nbox, 1029]
            #is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)
            is_min_metrics  = is_min_metrics.permute(0, 2, 1).to(metrics.dtype)
            
            # [B, nbox, 1029] 
            #mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
            mask_pos = torch.where(mask_multi_gts, is_min_metrics, mask_pos)

            # [B, 1029]
            fg_mask = mask_pos.sum(-2)
            
        # find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)

        # assigned target labels, (b, 1)
        batch_index = torch.arange(end=self.bs,
                                   dtype=torch.int64,
                                   device=true_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_index * self.num_max_boxes

        # [B, 1029, nclass]
        target_labels = true_labels.long().view(-1, true_labels.shape[-1])[target_gt_idx]

        # assigned target boxes
        target_bboxes = true_bboxes.view(-1, 4)[target_gt_idx]

        # assigned target scores
        target_labels.clamp(0)
        #target_scores = one_hot(target_labels, self.nc)
        target_scores = target_labels
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_bboxes, target_scores, fg_mask.bool()

    @staticmethod
    def df_loss(pred_dist, target):
        # Return sum of left and right DFL losses
        # Distribution Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        l_loss = cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape)
        r_loss = cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape)
        return (l_loss * wl + r_loss * wr).mean(-1, keepdim=True)

    @staticmethod
    def iou(box1, box2, eps=1e-7):
        # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

        # Intersection area
        area1 = b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)
        area2 = b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
        intersection = area1.clamp(0) * area2.clamp(0)

        # Union Area
        union = w1 * h1 + w2 * h2 - intersection + eps

        # IoU
        iou = intersection / union
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        # Complete IoU https://arxiv.org/abs/1911.08287v1
        c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
        # center dist ** 2
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
        # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
        v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - (rho2 / c2 + v * alpha)  # CIoU

def build_loss(model, config):
    loss_type = config['loss']

    if loss_type == 'tal':
        return TAL(model, config)
    elif loss_type == 'simota':
        return SimOTA(model, config)
    elif loss_type == 'normal':
        return Normal(model, config)