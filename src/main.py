import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from io_utils import load_video_frames
from seg_utils import get_segmentation_model, predict_masks
from smooth_utils import smooth_masks_moving_average
from metric_utils import compute_temporal_stability
from vis_utils import visualize_comparison, plot_stability, plot_experiment_results

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    videos_dir = os.path.join(base_dir, 'videos')
    plots_dir = os.path.join(base_dir, 'plots')
    
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    video_path = os.path.join(videos_dir, 'video.mp4')
    
    print(f"Загрузка видео: {video_path}...")
    frames = load_video_frames(video_path)
    print(f"Загружено {len(frames)} кадров.")

    print("Инициализация модели...")
    model = get_segmentation_model()
    
    print("Генерация масок (инференс)...")
    # Используем низкий порог, чтобы получить шумные маски для демонстрации эффекта
    masks_orig, probs_orig = predict_masks(model, frames, threshold=0.3)
    
    avg_iou_orig, ious_orig = compute_temporal_stability(masks_orig)
    print(f"Исходная стабильность (Average IoU): {avg_iou_orig:.4f}")

    window_sizes = [3, 5, 7, 9, 11, 15]
    experiment_results = []
    best_iou = -1
    best_window = -1
    best_masks_smooth = None
    best_ious_smooth = None
    print("\nЗапуск эксперимента по подбору размера окна сглаживания...")
    for ws in window_sizes:
        print(f"  Обработка с window_size={ws}...")
        curr_masks_smooth, curr_probs_smooth = smooth_masks_moving_average(probs_orig, window_size=ws)
        curr_avg_iou, curr_ious = compute_temporal_stability(curr_masks_smooth)
        
        experiment_results.append(curr_avg_iou)
        print(f"    -> Stability: {curr_avg_iou:.4f}")
        
        if curr_avg_iou > best_iou:
            best_iou = curr_avg_iou
            best_window = ws
            best_masks_smooth = curr_masks_smooth
            best_ious_smooth = curr_ious

    print(f"\nЛучший результат: Window Size = {best_window} со стабильностью {best_iou:.4f}")

    print("Генерация визуализаций...")
    comparison_path = os.path.join(videos_dir, 'comparison.mp4')
    stability_plot_path = os.path.join(plots_dir, 'stability_plot.png')
    experiment_plot_path = os.path.join(plots_dir, 'experiment_plot.png')

    visualize_comparison(frames, masks_orig, best_masks_smooth, comparison_path)
    plot_stability(ious_orig, best_ious_smooth, stability_plot_path)
    plot_experiment_results(window_sizes, experiment_results, experiment_plot_path)
    
    print("Готово! Результаты сохранены:")
    print(f"- {comparison_path} (Видео сравнение)")
    print(f"- {stability_plot_path} (График стабильности по времени)")
    print(f"- {experiment_plot_path} (График зависимости стабильности от размера окна)")

if __name__ == "__main__":
    main()