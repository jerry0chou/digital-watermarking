import cv2
import numpy as np
import pywt
import ffmpeg


def parse_frame_rate(rate_str):
    """Convert frame rate string (e.g. '30/1' or '30') to float"""
    if '/' in rate_str:
        numerator, denominator = map(float, rate_str.split('/'))
        return numerator / denominator
    return float(rate_str)


def read_video(video_path):
    """Universal video reader that supports both H.264 and AV1"""
    probe = ffmpeg.probe(video_path)
    video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    codec_name = video_info['codec_name']
    width = int(video_info['width'])
    height = int(video_info['height'])

    # Parse frame rate properly
    frame_rate = parse_frame_rate(video_info.get('r_frame_rate', '30/1'))

    if codec_name == 'h264':
        # Use OpenCV for H.264
        frames = []
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return np.array(frames), frame_rate
    else:
        # Use ffmpeg for other codecs (including AV1)
        out, _ = (
            ffmpeg
            .input(video_path)
            .output('pipe:', format='rawvideo', pix_fmt='bgr24')
            .run(capture_stdout=True)
        )
        frames = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
        return frames, frame_rate


def write_video(frames, output_path, fps, codec='h264'):
    """Universal video writer that supports both H.264 and AV1"""
    height, width = frames[0].shape[:2]

    if codec == 'h264':
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()
    else:  # AV1
        process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='bgr24',
                   s='{}x{}'.format(width, height), framerate=fps)
            .output(output_path, vcodec='libaom-av1', crf=30, pix_fmt='yuv420p')
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
        for frame in frames:
            process.stdin.write(frame.astype(np.uint8).tobytes())
        process.stdin.close()
        process.wait()


##

def embed_watermark(frame, watermark, alpha=0.3):  # 增加 alpha 值使水印更明显
    """
    Embed watermark into frame's top-left corner
    """
    # 计算水印的目标大小 (视频高度和宽度的1/4)
    target_height = frame.shape[0] // 4
    target_width = frame.shape[1] // 4

    # 保持水印的宽高比
    watermark_aspect = watermark.shape[1] / watermark.shape[0]
    if watermark_aspect > 1:  # 宽图
        target_width = int(target_height * watermark_aspect)
    else:  # 高图
        target_height = int(target_width / watermark_aspect)

    # 调整水印大小
    watermark_resized = cv2.resize(watermark, (target_width, target_height))
    watermark_gray = cv2.cvtColor(watermark_resized, cv2.COLOR_BGR2GRAY)
    watermark_normalized = watermark_gray / 255.0

    # 创建与原始帧大小相同的水印掩码
    full_watermark = np.zeros((frame.shape[0], frame.shape[1]))
    full_watermark[:target_height, :target_width] = watermark_normalized

    # Convert frame to float32
    frame_float = np.float32(frame)

    # Apply DWT to each channel
    channels = cv2.split(frame_float)
    watermarked_channels = []

    for channel in channels:
        coeffs2 = pywt.dwt2(channel, 'haar')
        cA, (cH, cV, cD) = coeffs2

        # Get DWT of full watermark
        watermark_coeffs = pywt.dwt2(full_watermark, 'haar')
        watermark_cA, _ = watermark_coeffs

        # Embed watermark into the approximation coefficients
        cA_watermarked = cA + alpha * watermark_cA

        # Reconstruct the channel with the watermarked coefficients
        coeffs2_watermarked = (cA_watermarked, (cH, cV, cD))
        channel_watermarked = pywt.idwt2(coeffs2_watermarked, 'haar')

        # Handle any None returns from idwt2
        if channel_watermarked is None:
            channel_watermarked = channel

        # Ensure the reconstructed channel matches the original size
        if channel_watermarked.shape != channel.shape:
            channel_watermarked = cv2.resize(channel_watermarked, (channel.shape[1], channel.shape[0]))

        watermarked_channels.append(channel_watermarked)

    # Merge channels and convert back to uint8
    watermarked_frame = cv2.merge(watermarked_channels)
    watermarked_frame = np.clip(watermarked_frame, 0, 255)
    watermarked_frame_uint8 = np.uint8(watermarked_frame)

    return watermarked_frame_uint8


def extract_watermark(frame_original, frame_watermarked, alpha=0.3):  # 增加 alpha 值保持一致
    """
    Extract watermark from watermarked frame
    """
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

    # 裁剪出左上角区域 (1/4 大小)
    height = watermark_extracted.shape[0]
    width = watermark_extracted.shape[1]
    watermark_extracted = watermark_extracted[:height // 4, :width // 4]

    watermark_extracted = np.clip(watermark_extracted, 0, 255)
    watermark_extracted_uint8 = np.uint8(watermark_extracted)

    return watermark_extracted_uint8


def embed_watermark_in_video(input_video_path, output_video_path, watermark_image_path, output_codec='h264'):
    """Embed watermark in video with support for both H.264 and AV1"""
    frames, fps = read_video(input_video_path)
    watermark = cv2.imread(watermark_image_path)

    if watermark is None:
        raise ValueError(f"Could not read watermark image from {watermark_image_path}")

    watermarked_frames = []

    print(f"Processing {len(frames)} frames at {fps} fps...")
    print(f"Video size: {frames[0].shape[1]}x{frames[0].shape[0]}")
    print(f"Watermark size: {watermark.shape[1]}x{watermark.shape[0]}")

    for i, frame in enumerate(frames):
        watermarked_frame = embed_watermark(frame, watermark, alpha=0.3)  # 增加 alpha 值
        watermarked_frames.append(watermarked_frame)
        if i % 10 == 0:  # Progress update every 10 frames
            print(f"Processed {i}/{len(frames)} frames")

    write_video(watermarked_frames, output_video_path, fps, codec=output_codec)
    print(f"Watermarked video saved as {output_video_path}")


def extract_watermark_from_video(original_video_path, watermarked_video_path, output_watermark_path):
    """Extract watermark from video with support for both H.264 and AV1"""
    frames_orig, _ = read_video(original_video_path)
    frames_wm, _ = read_video(watermarked_video_path)
    watermark_frames = []

    print(f"Processing {len(frames_orig)} frames...")

    for i, (frame_orig, frame_wm) in enumerate(zip(frames_orig, frames_wm)):
        watermark_frame = extract_watermark(frame_orig, frame_wm, alpha=0.3)  # 增加 alpha 值
        watermark_frames.append(watermark_frame)
        if i % 10 == 0:  # Progress update every 10 frames
            print(f"Processed {i}/{len(frames_orig)} frames")

    watermark_average = np.mean(watermark_frames, axis=0)
    watermark_average_uint8 = np.uint8(np.clip(watermark_average, 0, 255))
    cv2.imwrite(output_watermark_path, watermark_average_uint8)
    print(f"Extracted watermark saved as {output_watermark_path}")
##


if __name__ == "__main__":
    input_video = "input_video.mkv"
    watermarked_video = "watermarked_video.mkv"
    watermark_image = "watermark.png"
    extracted_watermark = "extracted_watermark.png"

    # Embed watermark (use h264 or av1 for output_codec)
    embed_watermark_in_video(input_video, watermarked_video, watermark_image, output_codec='h264')

    # Extract watermark 报错 未解决
    # extract_watermark_from_video(input_video, watermarked_video, extracted_watermark)