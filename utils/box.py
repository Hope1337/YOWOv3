import torch 
import time
import torchvision
import cv2
import numpy as np

def make_anchors(x, strides, offset=0.5):
    """
    Generate anchors from features
    """
    assert x is not None
    anchor_points, stride_tensor = [], []
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, dtype=x[i].dtype, device=x[i].device) + offset  # shift x
        sy = torch.arange(end=h, dtype=x[i].dtype, device=x[i].device) + offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=x[i].dtype, device=x[i].device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def wh2xy(x):
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # intersection(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    intersection = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = intersection / (area1 + area2 - intersection)
    box1 = box1.T
    box2 = box2.T

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return intersection / (area1[:, None] + area2 - intersection)

def non_max_suppression(prediction, conf_threshold=0.25, iou_threshold=0.45):
    nc = prediction.shape[1] - 4  # number of classes
    xc = prediction[:, 4:4 + nc].amax(1) > conf_threshold  # candidates

    # Settings
    max_wh = 7680  # (pixels) maximum box width and height
    max_det = 300  # the maximum number of boxes to keep after NMS
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()

    start = time.time()
    outputs = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for index, x in enumerate(prediction):  # image index, image inference
        # Apply constraints

        # [n, 84]
        x = x.transpose(0, -1)[xc[index]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (box, conf, cls)
        box, cls = x.split((4, nc), 1)
        # center_x, center_y, width, height) to (x1, y1, x2, y2)
        box = wh2xy(box)
        if nc > 1:
            i, j = (cls > conf_threshold).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_threshold]
        # Check shape
        if not x.shape[0]:  # no boxes
            continue
        # sort by confidence and remove excess boxes
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_threshold)  # NMS
        i = i[:max_det]  # limit detections
        outputs[index] = x[i]
        #print(x[i])
        if (time.time() - start) > 0.5 + 0.05 * prediction.shape[0]:
            print(f'WARNING ⚠️ NMS time limit {0.5 + 0.05 * prediction.shape[0]:.3f}s exceeded')
            break  # time limit exceeded

    return outputs

def opacity(img, pt1, pt2):
    x = pt1[0]
    y = pt1[1]
    w = pt2[0] - pt1[0]
    h = pt2[1] - pt1[1] 
    sub_img = img[y:y+h, x:x+w]
    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255

    res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)

    # Putting the image back to its position
    img[y:y+h, x:x+w] = res

# adapted from https://inside-machinelearning.com/en/bounding-boxes-python-function/
# một chút sửa đổi để phù hợp với mục đích sử dụng
def box_label(image, box, label=None, color=(0, 255, 0), txt_color=(0, 0, 0)):
  """
  :param image : np array [H, W, C] (BGR)
  :param label : text, default = None
  """
  lw = max(round(sum(image.shape) / 2 * 0.003), 2)
  p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
  cv2.rectangle(image, p1, p2, color, thickness=lw-1, lineType=cv2.LINE_AA)
  if label is not None:
    tf = max(lw - 1, 1)  # font thickness
    w, h = cv2.getTextSize(label, 0, fontScale=lw / 10, thickness=tf)[0]  # text width, height
    outside = p1[1] - h >= 3
    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    y0, dy = 0,10
    y0 = p1[1] - 2 if outside else p1[1] + h + 2
    for i, line in enumerate(label.split('\n')):
        y = y0 + i*dy
        x = p1[0]
        text_size, _ = cv2.getTextSize(line, cv2.LINE_AA, lw/5, tf)
        text_width, text_height = text_size
        # Vẽ hình chữ nhật bao quanh văn bản
        rectangle_color = (0, 255, 0)  # Màu xanh lá cây
        rectangle_thickness = 1
        #cv2.rectangle(image, (x, y), (x + text_width + 1, y + text_height + 1), rectangle_color, cv2.FILLED, cv2.LINE_AA)
        opacity(image, (x, y), (x + text_width + 1, y + text_height + 1))
        cv2.putText(image,
                line, (x, y + text_height),
                0,
                lw / 5,
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA)

    
def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # intersection(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    intersection = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = intersection / (area1 + area2 - intersection)
    box1 = box1.T
    box2 = box2.T

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return intersection / (area1[:, None] + area2 - intersection)

def draw_bounding_box(image, bboxes, labels, confs, map_labels):
    """
    Vẽ bouding box, đầu và image có thể là tensor hoặc np array tuỳ tình huống, đã thiết kế để xử lý cả 2 trường hợp
    
    :param image, tensor [1, 3, H, W] (RGB) hoặc numpy array [H, W, 3] (BGR)
    :param bboxes, tensor [nbox, 4]
    :param labels, tensor [nbox]
    :param confs, tensor [nbox]
    :param map_labels, dict, do labels là số nên cần map thành nhãn (string)
    """
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image.squeeze(0)
        image = image.permute(1, 2, 0)[:, :, (2, 1, 0)].contiguous()
        image = image.detach().cpu().numpy()
    
    H, W, C = image.shape 

    pre_box   = []
    meta_data = []

    if bboxes is not None:
        for box, label, conf in zip(bboxes, labels, confs):
            box = box.clone().detach()
            #box[0] = max(0, int(box[0]))
            #box[1] = max(0, int(box[1]))
            #box[2] = min(W, int(box[2]))
            #box[3] = min(H, int(box[3]))
            text    = str(map_labels[int(label.item())] + " : " + str(round(conf.item()*100, 2)))
            #box_label(image, box, text)
            pre_box.append(box)
            meta_data.append([text, H, W])

        res_box = []
        res_meta_data = []

        for idx1, box1 in enumerate(pre_box):
            flag = 1
            for idx2, box2 in enumerate(res_box):
                iou = box_iou(box1.unsqueeze(0), box2.unsqueeze(0))[0, 0]
                if (iou >= 0.9):
                    res_meta_data[idx2].append(meta_data[idx1])
                    flag = 0
                    break
            if flag:
                res_box.append(box1)
                res_meta_data.append([meta_data[idx1]])

        for box, meta in zip(res_box, res_meta_data):
            text = ''
            for sub_meta in meta:
                if text == '':
                    text = sub_meta[0]
                else:
                    text += '\n' + sub_meta[0]

            H = meta[0][1]
            W = meta[0][2]
            
            bbox = []

            bbox.append(int(max(0, box[0])))
            bbox.append(int(max(0, box[1])))
            bbox.append(int(min(W, box[2])))
            bbox.append(int(min(H, box[3])))

            box_label(image, bbox, text)






    