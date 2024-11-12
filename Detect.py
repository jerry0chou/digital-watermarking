import Watermark

input_video = "967_raw.mp4" 
watermarked_video = "967_raw_watermarked.mp4"
key = 12345                   # Key for watermark generation
alpha = 5.0   

# Detect the watermark in the watermarked video
Watermark.detect_watermark_in_video(input_video, watermarked_video, key=key, alpha=alpha, threshold=0.0)