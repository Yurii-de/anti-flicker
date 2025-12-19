import numpy as np

def smooth_masks_moving_average(prob_masks, window_size=5):
    """
    Сглаживает вероятностные маски, используя временное скользящее среднее.
    
    Аргументы:
        prob_masks: Список вероятностных масок (H, W) со значениями [0, 1].
        window_size: Размер окна сглаживания (должен быть нечетным).
        
    Возвращает:
        smoothed_binary_masks: Список бинарных масок после сглаживания.
        smoothed_prob_masks: Список вероятностных масок после сглаживания.
    """
    if window_size % 2 == 0:
        window_size += 1
        
    pad_size = window_size // 2
    n_frames = len(prob_masks)
    h, w = prob_masks[0].shape
    
    masks_stack = np.stack(prob_masks, axis=0)
    smoothed_probs = np.zeros_like(masks_stack)
    
    print(f"Сглаживание с размером окна {window_size}...")
    for i in range(n_frames):
        start = max(0, i - pad_size)
        end = min(n_frames, i + pad_size + 1)
        
        smoothed_probs[i] = np.mean(masks_stack[start:end], axis=0)
        
    smoothed_binary = (smoothed_probs > 0.5).astype(np.uint8)
    return list(smoothed_binary), list(smoothed_probs)