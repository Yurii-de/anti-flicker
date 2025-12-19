import cv2
import numpy as np

def load_video_frames(video_path):
    """
    Загружает все кадры из видеофайла.
    Возвращает список numpy массивов (RGB).
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames

def save_video(frames, output_path, fps=30):
    """
    Сохраняет список кадров (RGB) в видеофайл.
    """
    if not frames:
        return
    
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()