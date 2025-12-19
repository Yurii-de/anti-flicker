import numpy as np

def compute_iou(mask1, mask2):
    """
    Вычисляет Intersection over Union (IoU) между двумя бинарными масками.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def compute_temporal_stability(masks):
    """
    Вычисляет средний IoU между соседними кадрами.
    Более высокий IoU означает большую стабильность (при условии, что объект не движется слишком быстро).
    """
    ious = []
    for i in range(len(masks) - 1):
        iou = compute_iou(masks[i], masks[i+1])
        ious.append(iou)
    
    if not ious:
        return 0.0
    
    return np.mean(ious), ious