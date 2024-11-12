import cv2
import numpy as np
import pywt
import os
import subprocess
import matplotlib.pyplot as plt

from utils import isWindows

FFMPEG = 'ffmpeg'
if isWindows():
    FFMPEG = 'G:\\Temp\\ffmpeg\\bin\\ffmpeg.exe'

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

    # Initialize extracted watermark components
    extracted_components = []

    # Extract watermark from each channel
    for i in range(3):  # Assuming BGR channels
        # DWT on original frame
        coeffs2_orig = pywt.dwt2(original_float[:, :, i], 'haar')
        _, (cH_orig, cV_orig, cD_orig) = coeffs2_orig

        # DWT on watermarked frame
        coeffs2_wm = pywt.dwt2(watermarked_float[:, :, i], 'haar')
        _, (cH_wm, cV_wm, cD_wm) = coeffs2_wm

        # Extract the watermark components
        wH_extracted = (cH_wm - cH_orig) / alpha
        wV_extracted = (cV_wm - cV_orig) / alpha
        wD_extracted = (cD_wm - cD_orig) / alpha

        # Average the extracted watermark components
        watermark_extracted = (wH_extracted + wV_extracted + wD_extracted) / 3.0

        # Ensure the watermark is real-valued
        watermark_extracted = np.real(watermark_extracted)

        # Debugging statements
        #print(f"Channel {i}:")
        #print(f"Type of watermark_extracted: {type(watermark_extracted)}")
        #print(f"Shape of watermark_extracted: {watermark_extracted.shape}")

        extracted_components.append(watermark_extracted)

    # Average over all channels
    extracted_watermark = sum(extracted_components) / 3.0

    # Ensure the extracted watermark is real-valued
    extracted_watermark = np.real(extracted_watermark)

    # Debugging statements
    #print(f"Type of extracted_watermark: {type(extracted_watermark)}")
    #print(f"Shape of extracted_watermark: {extracted_watermark.shape}")

    return extracted_watermark

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

    ffmpeg_path = FFMPEG
    # Check if input files exist
    if not os.path.isfile(original_video_path):
        print(f"Original video file not found: {original_video_path}")
        return

    if not os.path.isfile(video_no_audio_path):
        print(f"Video without audio file not found: {video_no_audio_path}")
        return

    command = [
        ffmpeg_path,
        '-y',  # Overwrite output files without asking
        '-i', video_no_audio_path,  # Input: video without audio
        '-i', original_video_path,  # Input: original video with audio
        '-c', 'copy',  # Copy both video and audio streams without re-encoding
        '-map', '0:v?',  # Map video stream from the first input (watermarked video) ? stands for none v
        '-map', '1:a?',  # Map audio stream from the second input (original video) ? stands for none a
        output_video_path  # Output file
    ]
    print('Running FFmpeg command:', ' '.join(command))
    # Run the command
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("FFmpeg command failed with the following error:")
        print(result.stderr)
    else:
        print("Audio successfully added to the video.")

def detect_watermark_in_video(original_video_path, watermarked_video_path, key, alpha=0.1, threshold=0.0):
    cap_orig = cv2.VideoCapture(original_video_path)
    cap_wm = cv2.VideoCapture(watermarked_video_path)

    total_correlation = 0.0  # Use a float scalar
    frame_count = 0

    while True:
        ret_orig, frame_orig = cap_orig.read()
        ret_wm, frame_wm = cap_wm.read()

        if not ret_orig or not ret_wm:
            break

        correlation = compute_correlation(frame_orig, frame_wm, key, alpha=alpha)
        total_correlation += correlation
        frame_count += 1

        # Debugging statements
        #print(f"Frame {frame_count}: correlation={correlation}")

    cap_orig.release()
    cap_wm.release()

    if frame_count == 0:
        print("No frames were processed.")
        return

    # Average correlation over all frames
    average_correlation = total_correlation / frame_count

    # Ensure average_correlation is a scalar
    average_correlation = float(average_correlation)

    print(f"Average Correlation: {average_correlation}")

    # Decide whether the watermark is present
    if average_correlation > threshold:
        print("Watermark detected.")
    else:
        print("Watermark not detected or video has been tampered with.")

def compute_correlation(frame_original, frame_watermarked, key, alpha=0.1):
    # Convert frames to float32
    original_float = np.float32(frame_original)
    watermarked_float = np.float32(frame_watermarked)

    # Initialize correlation sum
    correlation = 0.0  # Use a float scalar

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

        # Resize watermark_normalized to match coefficient sizes
        watermark_normalized = cv2.resize(watermark_normalized, (cH_orig.shape[1], cH_orig.shape[0]))

        # Extract the watermark components
        wH_extracted = (cH_wm - cH_orig) / alpha
        wV_extracted = (cV_wm - cV_orig) / alpha
        wD_extracted = (cD_wm - cD_orig) / alpha

        # Calculate correlation for each component
        corr_wH = np.sum(np.real(wH_extracted * watermark_normalized))
        corr_wV = np.sum(np.real(wV_extracted * watermark_normalized))
        corr_wD = np.sum(np.real(wD_extracted * watermark_normalized))

        # Debugging statements
        #print(f"Channel {i} correlations: wH={corr_wH}, wV={corr_wV}, wD={corr_wD}")

        # Accumulate the correlations
        correlation += corr_wH + corr_wV + corr_wD

    # Ensure correlation is a real scalar
    correlation = np.real(correlation).item()

    return correlation

def extract_and_save_watermark(original_video_path, watermarked_video_path, key, alpha=0.1):
    cap_orig = cv2.VideoCapture(original_video_path)
    cap_wm = cv2.VideoCapture(watermarked_video_path)

    frame_count = 0
    accumulated_watermark = None

    while True:
        ret_orig, frame_orig = cap_orig.read()
        ret_wm, frame_wm = cap_wm.read()

        if not ret_orig or not ret_wm:
            break

        extracted_watermark = extract_watermark(frame_orig, frame_wm, key, alpha=alpha)

        if accumulated_watermark is None:
            accumulated_watermark = np.zeros_like(extracted_watermark, dtype=np.float64)

        accumulated_watermark += extracted_watermark
        frame_count += 1

    cap_orig.release()
    cap_wm.release()

    if frame_count == 0:
        print("No frames were processed.")
        return

    # Average the accumulated watermark over all frames
    average_watermark = accumulated_watermark / frame_count

    # Ensure the average watermark is real-valued
    average_watermark = np.real(average_watermark)

    # Normalize the extracted watermark to enhance visibility
    min_val = np.min(average_watermark)
    max_val = np.max(average_watermark)
    print(f"Extracted watermark min value: {min_val}, max value: {max_val}")

    # Avoid division by zero
    if max_val - min_val > 0:
        normalized_watermark = (average_watermark - min_val) / (max_val - min_val)
        normalized_watermark = normalized_watermark * 255
    else:
        normalized_watermark = average_watermark

    normalized_watermark_uint8 = np.uint8(np.clip(normalized_watermark, 0, 255))

    # Save the extracted watermark image
    base, _ = os.path.splitext(watermarked_video_path)
    output_watermark_path = f"{base}_extracted_watermark.png"
    cv2.imwrite(output_watermark_path, normalized_watermark_uint8)
    print(f"Extracted watermark saved as {output_watermark_path}")

    plt.imshow(normalized_watermark_uint8, cmap='gray')
    plt.title('Extracted Watermark')
    plt.show()

