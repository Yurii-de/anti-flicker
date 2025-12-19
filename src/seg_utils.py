import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
import cv2
import numpy as np
from tqdm import tqdm

def get_segmentation_model():
    """
    Возвращает предобученную модель Mask R-CNN.
    """
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')
    return model

def predict_masks(model, frames, threshold=0.5):
    """
    Запускает инференс на списке кадров и возвращает бинарные маски.
    Для простоты мы берем маску объекта с самой высокой уверенностью.
    
    Возвращает:
        masks: Список бинарных масок (H, W) со значениями 0 или 1.
        probs: Список карт вероятностей (H, W) со значениями [0, 1].
    """
    device = next(model.parameters()).device
    binary_masks = []
    prob_masks = []
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    
    print("Запуск сегментации...")
    with torch.no_grad():
        for frame in tqdm(frames):
            img_tensor = transform(frame).to(device)
            predictions = model([img_tensor])[0]
            
            scores = predictions['scores'].cpu().numpy()
            masks = predictions['masks'].cpu().numpy()
            
            if len(scores) > 0 and scores[0] > threshold:
                best_mask = masks[0, 0]
                prob_masks.append(best_mask)
                binary_masks.append((best_mask > 0.5).astype(np.uint8))
            else:
                h, w = frame.shape[:2]
                prob_masks.append(np.zeros((h, w), dtype=np.float32))
                binary_masks.append(np.zeros((h, w), dtype=np.uint8))
                
    return binary_masks, prob_masks