import cv2
import numpy as np
import pywt
import os
import subprocess

def generate_watermark_pattern(size, key):
    np.random.seed(key)  # Use a seed for reproducibility
    watermark = np.random.rand(size[1], size[0]) * 255  # Generate random noise pattern
    watermark_uint8 = np.uint8(watermark)
    return watermark_uint8

def embed_watermark(frame, watermark, alpha=0.1):
    # Convert frame to float32
    frame_float = np.float32(frame)

    # Apply DWT to each channel
    channels = cv2.split(frame_float)
    watermarked_channels = []
    for channel in channels:
        coeffs2 = pywt.dwt2(channel, 'haar')
        cA, (cH, cV, cD) = coeffs2

        # Resize watermark to match the size of cH
        watermark_resized = cv2.resize(watermark, (cH.shape[1], cH.shape[0]))
        watermark_normalized = watermark_resized / 255.0

        # Embed watermark into the detail coefficients
        cH_watermarked = cH + alpha * watermark_normalized
        cV_watermarked = cV + alpha * watermark_normalized
        cD_watermarked = cD + alpha * watermark_normalized

        # Reconstruct the channel with the watermarked coefficients
        coeffs2_watermarked = (cA, (cH_watermarked, cV_watermarked, cD_watermarked))
        channel_watermarked = pywt.idwt2(coeffs2_watermarked, 'haar')
        watermarked_channels.append(channel_watermarked)

    # Merge channels and convert back to uint8
    watermarked_frame = cv2.merge(watermarked_channels)
    watermarked_frame = np.clip(watermarked_frame, 0, 255)
    watermarked_frame_uint8 = np.uint8(watermarked_frame)

    return watermarked_frame_uint8

def extract_watermark(frame_original, frame_watermarked, key, alpha=0.1):
    # Convert frames to float32
    original_float = np.float32(frame_original)
    watermarked_float = np.float32(frame_watermarked)

    # Initialize correlation sum
    correlation = 0

    # Extract watermark from each channel
    for i in range(3):  # Assuming BGR channels
        # DWT on original frame
        coeffs2_orig = pywt.dwt2(original_float[:, :, i], 'haar')
        _, (cH_orig, cV_orig, cD_orig) = coeffs2_orig

        # DWT on watermarked frame
        coeffs2_wm = pywt.dwt2(watermarked_float[:, :, i], 'haar')
        _, (cH_wm, cV_wm, cD_wm) = coeffs2_wm

        # Generate the same watermark pattern using the key
        watermark_size = (cH_orig.shape[1], cH_orig.shape[0])
        watermark = generate_watermark_pattern(watermark_size, key)
        watermark_normalized = watermark / 255.0

        # Extract the watermark components
        wH_extracted = (cH_wm - cH_orig) / alpha
        wV_extracted = (cV_wm - cV_orig) / alpha
        wD_extracted = (cD_wm - cD_orig) / alpha

        # Calculate correlation
        correlation += np.sum(wH_extracted * watermark_normalized)
        correlation += np.sum(wV_extracted * watermark_normalized)
        correlation += np.sum(wD_extracted * watermark_normalized)

    return correlation

def embed_watermark_in_video(input_video_path, key, alpha=0.1):
    # Generate output video path by appending '_watermarked' to the input filename
    base, ext = os.path.splitext(input_video_path)
    temp_video_path = f"{base}_watermarked_temp{ext}"  # Temporary video without audio
    output_video_path = f"{base}_watermarked{ext}"     # Final video with audio

    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Use 'mp4v' codec for MP4 files or 'XVID' for AVI files
    if ext.lower() in ['.mp4', '.m4v']:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif ext.lower() in ['.avi']:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    else:
        # Default codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (frame_width, frame_height))

    # Generate watermark pattern based on the frame size and key
    # Note: We don't resize it here anymore
    watermark = generate_watermark_pattern((frame_width, frame_height), key)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        watermarked_frame = embed_watermark(frame, watermark, alpha=alpha)
        out.write(watermarked_frame)

    cap.release()
    out.release()

    # Now, combine the watermarked video with the original audio
    add_audio_to_video(input_video_path, temp_video_path, output_video_path)

    # Remove the temporary video file
    os.remove(temp_video_path)

    print(f"Watermarked video saved as {output_video_path}")

def add_audio_to_video(original_video_path, video_no_audio_path, output_video_path):
    # Use FFmpeg to extract audio and combine with watermarked video

    ffmpeg_path = "G:\\Temp\\ffmpeg\\bin\\ffmpeg.exe"

    if not os.path.isfile(ffmpeg_path):
        print(f"FFmpeg not found at the specified path: {ffmpeg_path}")
        return
    
    command = [
        ffmpeg_path,
        '-y',  # Overwrite output files without asking
        '-i', video_no_audio_path,
        '-i', original_video_path,
        '-c', 'copy',
        '-map', '0:v:0',
        '-map', '1:a:0',
        output_video_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("FFmpeg command failed with the following error:")
        print(result.stderr)
    else:
        print("Audio successfully added to the video.")

def detect_watermark_in_video(original_video_path, watermarked_video_path, key, alpha=0.1, threshold=0.0):
    cap_orig = cv2.VideoCapture(original_video_path)
    cap_wm = cv2.VideoCapture(watermarked_video_path)

    total_correlation = 0
    frame_count = 0

    while True:
        ret_orig, frame_orig = cap_orig.read()
        ret_wm, frame_wm = cap_wm.read()

        if not ret_orig or not ret_wm:
            break

        correlation = extract_watermark(frame_orig, frame_wm, key, alpha=alpha)
        total_correlation += correlation
        frame_count += 1

    cap_orig.release()
    cap_wm.release()

    if frame_count == 0:
        print("No frames were processed.")
        return

    # Average correlation over all frames
    average_correlation = total_correlation / frame_count

    print(f"Average Correlation: {average_correlation}")

    # Decide whether the watermark is present
    if average_correlation > threshold:
        print("Watermark detected.")
    else:
        print("Watermark not detected or video has been tampered with.")



if __name__ == "__main__":
    input_video = "G:\\Temp\\Output\\967_raw.mp4"  # Your input video file (e.g., MP4, AVI)
    key = 12345                   # Key for watermark generation (choose any integer)

    # Embed the invisible watermark into the video
    embed_watermark_in_video(input_video, key=key, alpha=0.1)

    # Generate watermarked video filename
    base, ext = os.path.splitext(input_video)
    watermarked_video = f"{base}_watermarked{ext}"

    # Detect the watermark in the watermarked video
    detect_watermark_in_video(input_video, watermarked_video, key=key, alpha=0.1, threshold=0.0)

