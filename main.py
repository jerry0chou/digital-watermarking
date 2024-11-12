# Import the watermarking module (assuming it's named 'video_watermarking.py')
import Watermark
import os

# Set your input video path and parameters
input_video = "G:\\Temp\\Output\\967_raw.mp4" 
key = 12345                   # Key for watermark generation
alpha = 5.0                   # Embedding strength (adjust as needed)

# Embed the watermark into the video
Watermark.embed_watermark_in_video(input_video, key=key, alpha=alpha)

# Generate watermarked video
base, ext = os.path.splitext(input_video)
watermarked_video = f"{base}_watermarked{ext}"

# Detect the watermark in the watermarked video
Watermark.detect_watermark_in_video(input_video, watermarked_video, key=key, alpha=alpha, threshold=0.0)

# Extract and save the embedded watermark as an image
Watermark.extract_and_save_watermark(input_video, watermarked_video, key=key, alpha=alpha)
