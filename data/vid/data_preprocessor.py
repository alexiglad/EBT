import numpy as np
import cv2
import time
from PIL import Image
import torch

def remove_padding(image, threshold_value=30, threshold_ratio=0.8):
    h, w = image.shape[:2]
    row_sum = np.sum(np.all(image <= threshold_value, axis=-1), axis=1)
    col_sum = np.sum(np.all(image <= threshold_value, axis=-1), axis=0)
    
    row_indices = np.where(row_sum < threshold_ratio * w)[0]
    col_indices = np.where(col_sum < threshold_ratio * h)[0]
    
    # Default to the full range if indices are empty
    row_start = row_indices[0] if row_indices.size != 0 else 0
    row_end = row_indices[-1] if row_indices.size != 0 else h - 1
    col_start = col_indices[0] if col_indices.size != 0 else 0
    col_end = col_indices[-1] if col_indices.size != 0 else w - 1
    
    cropped = False  # Flag to indicate if cropping occurred
    
    # Check if the image dimensions have changed due to cropping
    if row_start > 0 or row_end < h - 1 or col_start > 0 or col_end < w - 1:
        cropped = True
    
    cropped_image = image[row_start:row_end+1, col_start:col_end+1]
    
    return cropped_image, cropped

def crop_to_square(image):
    h, w = image.shape[:2]
    min_dim = min(h, w)
    
    # Calculate the cropping coordinates
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    
    # Crop the image
    square_image = image[start_y:start_y + min_dim, start_x:start_x + min_dim]
    
    return square_image

def get_all_frames(video_path, hparams, transform):
    x = time.time()
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_height, original_width = frame.shape[:2]
        if hparams.preprocess_data:
            frame, cropped = remove_padding(original_frame)  # Assuming this function is defined
            if frame.shape[0] < original_height / 2 or frame.shape[1] < original_width / 2:
                # Call remove_padding again with different threshold values
                frame, _ = remove_padding(frame, threshold_value=10, threshold_ratio=0.9)
                if frame.shape[0] < original_height / 2 or frame.shape[1] < original_width / 2:
                    frame, _ = remove_padding(frame, threshold_value=10, threshold_ratio=0.95)
            
            if hparams.crop_all_samples or cropped:
                frame = crop_to_square(frame)  # Assuming this function is defined, we crop all images generally since dont want non uniform distortion

        square_pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        transformed_frame = transform(square_pil_image)
        frames.append(transformed_frame)
    #print(f"get all frames took {time.time()-x} to run")
    
    return torch.stack(frames)

def handle_corrupt_file(exception, video_path, log_path):
        with open(log_path, 'a') as f:
            f.write(f"\n{video_path}")

        error_text = f"""
        Error on video: {video_path}
        The video path has been logged to {log_path} and will be ignored in future runs.
        Exception string: {str(exception)}
        """

        # raise Exception(error_text)