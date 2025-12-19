import cv2
import numpy as np
import matplotlib.pyplot as plt

def overlay_mask_on_frame(frame, mask, color=(0, 255, 0), alpha=0.5):
    """
    Накладывает бинарную маску на кадр.
    """
    mask_bool = mask > 0
    overlay = frame.copy()
    overlay[mask_bool] = color
    
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

def visualize_comparison(frames, masks_orig, masks_smooth, output_path):
    """
    Создает видео со сравнением бок-о-бок.
    """
    height, width, _ = frames[0].shape
    out_width = 2 * width
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (out_width, height))
    
    for i in range(len(frames)):
        frame = frames[i]
        m_orig = masks_orig[i]
        m_smooth = masks_smooth[i]
        
        vis_orig = overlay_mask_on_frame(frame, m_orig, color=(255, 0, 0))
        vis_smooth = overlay_mask_on_frame(frame, m_smooth, color=(0, 255, 0))
        
        cv2.putText(vis_orig, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis_smooth, "Smoothed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        combined = np.hstack((vis_orig, vis_smooth))
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        out.write(combined_bgr)
        
    out.release()

def plot_stability(ious_orig, ious_smooth, output_path='stability_plot.png'):
    """
    Строит график метрики стабильности IoU.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(ious_orig, label='Original', alpha=0.7)
    plt.plot(ious_smooth, label='Smoothed', alpha=0.7)
    plt.title('Временная стабильность (IoU между соседними кадрами)')
    plt.xlabel('Индекс кадра')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def plot_experiment_results(window_sizes, stabilities, output_path='experiment_plot.png'):
    """
    Строит график зависимости стабильности от размера окна.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(window_sizes, stabilities, marker='o', linestyle='-', color='b')
    plt.title('Влияние размера окна сглаживания на стабильность (IoU)')
    plt.xlabel('Размер окна (Window Size)')
    plt.ylabel('Средний IoU (Stability)')
    plt.grid(True)
    plt.xticks(window_sizes)
    plt.savefig(output_path)
    plt.close()