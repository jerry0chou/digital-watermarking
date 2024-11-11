import cv2
import numpy as np
import pywt

def embed_watermark(frame, watermark, alpha=0.1):
    # Resize watermark to match frame size
    watermark_resized = cv2.resize(watermark, (frame.shape[1], frame.shape[0]))
    watermark_gray = cv2.cvtColor(watermark_resized, cv2.COLOR_BGR2GRAY)
    watermark_normalized = watermark_gray / 255.0

    # Convert frame to float32
    frame_float = np.float32(frame)

    # Apply DWT to each channel
    channels = cv2.split(frame_float)
    watermarked_channels = []
    for channel in channels:
        coeffs2 = pywt.dwt2(channel, 'haar')
        cA, (cH, cV, cD) = coeffs2

        # Embed watermark into the approximation coefficients
        cA_watermarked = cA + alpha * watermark_normalized

        # Reconstruct the channel with the watermarked coefficients
        coeffs2_watermarked = (cA_watermarked, (cH, cV, cD))
        channel_watermarked = pywt.idwt2(coeffs2_watermarked, 'haar')
        watermarked_channels.append(channel_watermarked)

    # Merge channels and convert back to uint8
    watermarked_frame = cv2.merge(watermarked_channels)
    watermarked_frame = np.clip(watermarked_frame, 0, 255)
    watermarked_frame_uint8 = np.uint8(watermarked_frame)

    return watermarked_frame_uint8

def extract_watermark(frame_original, frame_watermarked, alpha=0.1):
    # Convert frames to float32
    original_float = np.float32(frame_original)
    watermarked_float = np.float32(frame_watermarked)

    # Initialize watermark
    watermark_extracted = np.zeros_like(original_float[:, :, 0])

    # Extract watermark from each channel
    for i in range(3):  # Assuming BGR channels
        # DWT on original frame
        coeffs2_orig = pywt.dwt2(original_float[:, :, i], 'haar')
        cA_orig, _ = coeffs2_orig

        # DWT on watermarked frame
        coeffs2_wm = pywt.dwt2(watermarked_float[:, :, i], 'haar')
        cA_wm, _ = coeffs2_wm

        # Extract the watermark
        watermark_channel = (cA_wm - cA_orig) / alpha
        watermark_extracted += watermark_channel

    # Average over channels
    watermark_extracted /= 3.0
    watermark_extracted = np.clip(watermark_extracted, 0, 255)
    watermark_extracted_uint8 = np.uint8(watermark_extracted)

    return watermark_extracted_uint8

def embed_watermark_in_video(input_video_path, output_video_path, watermark_image_path):
    cap = cv2.VideoCapture(input_video_path)
    watermark = cv2.imread(watermark_image_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    codec = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_video_path, codec, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        watermarked_frame = embed_watermark(frame, watermark, alpha=0.1)
        out.write(watermarked_frame)

    cap.release()
    out.release()
    print(f"Watermarked video saved as {output_video_path}")

def extract_watermark_from_video(original_video_path, watermarked_video_path, output_watermark_path):
    cap_orig = cv2.VideoCapture(original_video_path)
    cap_wm = cv2.VideoCapture(watermarked_video_path)

    watermark_frames = []

    while cap_orig.isOpened() and cap_wm.isOpened():
        ret_orig, frame_orig = cap_orig.read()
        ret_wm, frame_wm = cap_wm.read()
        if not ret_orig or not ret_wm:
            break

        watermark_frame = extract_watermark(frame_orig, frame_wm, alpha=0.1)
        watermark_frames.append(watermark_frame)

    # Average the extracted watermark frames to reduce noise
    watermark_average = np.mean(watermark_frames, axis=0)
    watermark_average_uint8 = np.uint8(np.clip(watermark_average, 0, 255))
    cv2.imwrite(output_watermark_path, watermark_average_uint8)
    print(f"Extracted watermark saved as {output_watermark_path}")

    cap_orig.release()
    cap_wm.release()