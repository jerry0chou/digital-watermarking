import cv2
import numpy as np
import pywt
import ffmpeg

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

def read_av1_video(video_path):
    probe = ffmpeg.probe(video_path)
    video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    out, _ = (
        ffmpeg
        .input(video_path)
        .output('pipe:', format='rawvideo', pix_fmt='bgr24')
        .run(capture_stdout=True)
    )
    video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
    return video

def write_av1_video(frames, output_video_path, fps):
    height, width = frames[0].shape[:2]
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(width, height), framerate=fps)
        .output(output_video_path, vcodec='libaom-av1', crf=30, pix_fmt='yuv420p')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    for frame in frames:
        process.stdin.write(frame.astype(np.uint8).tobytes())
    process.stdin.close()
    process.wait()

# The embed_watermark and extract_watermark functions remain the same

def embed_watermark_in_av1(input_video_path, output_video_path, watermark_image_path):
    frames = read_av1_video(input_video_path)
    watermark = cv2.imread(watermark_image_path)
    watermarked_frames = []

    for frame in frames:
        watermarked_frame = embed_watermark(frame, watermark, alpha=0.1)
        watermarked_frames.append(watermarked_frame)

    fps = 30  # Set the FPS accordingly or extract from the original video
    write_av1_video(watermarked_frames, output_video_path, fps)
    print(f"Watermarked video saved as {output_video_path}")

def extract_watermark_from_av1(original_video_path, watermarked_video_path, output_watermark_path):
    frames_orig = read_av1_video(original_video_path)
    frames_wm = read_av1_video(watermarked_video_path)
    watermark_frames = []

    for frame_orig, frame_wm in zip(frames_orig, frames_wm):
        watermark_frame = extract_watermark(frame_orig, frame_wm, alpha=0.1)
        watermark_frames.append(watermark_frame)

    # Average the extracted watermark frames to reduce noise
    watermark_average = np.mean(watermark_frames, axis=0)
    watermark_average_uint8 = np.uint8(np.clip(watermark_average, 0, 255))
    cv2.imwrite(output_watermark_path, watermark_average_uint8)
    print(f"Extracted watermark saved as {output_watermark_path}")


if __name__ == "__main__":
    input_video = "input_av1_video.mkv"  # Your AV1-encoded input video
    watermarked_video = "watermarked_av1_video.mkv"
    watermark_image = "watermark.png"
    extracted_watermark = "extracted_watermark.png"

    # Embed watermark
    embed_watermark_in_video(input_video, watermarked_video, watermark_image)

    # Extract watermark
    extract_watermark_from_video(input_video, watermarked_video, extracted_watermark)