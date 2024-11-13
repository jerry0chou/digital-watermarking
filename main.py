# Import the watermarking module (assuming it's named 'video_watermarking.py')
import Watermark
import os
import click
from utils import isWindows

# Set your input video path and parameters
input_video = "star.mp4"
if isWindows():
    input_video  = 'G:\\Temp\\Output\\967_raw.mp4'
key = 12345                   # Key for watermark generation
alpha = 5.0                   # Embedding strength (adjust as needed)

def menu():
    # 设置框的宽度（确保足够容纳最长的行）
    width = 55

    # 定义菜单内容，使用 .ljust(width - 4) 保持对齐
    menu_text = f'''
{click.style("+" + "-"*(width-2) + "+", fg="cyan")}
{click.style("+ 0: exit program".ljust(width-1) + "+", fg="yellow")}
{click.style("+ 1: embed watermark into video".ljust(width-1) + "+", fg="yellow")}
{click.style("+ 2: compare original video with watermarked video".ljust(width-1) + "+", fg="yellow")}
{click.style("+ 3: compare original video with duplicated video".ljust(width-1) + "+", fg="yellow")}
{click.style("+ 4: extract watermark".ljust(width-1) + "+", fg="yellow")}
{click.style("+" + "-"*(width-2) + "+", fg="cyan")}
    '''
    # 打印美化后的菜单
    click.echo(menu_text)

def cyan_print(str):
    click.echo(click.style(str, fg="cyan"))

if __name__ == '__main__':
    while True:
        try:
            menu()
            # cyan_print("Please enter a number:\n")
            num = click.prompt(click.style("Please enter a number", fg="cyan"), type=int)  # 输入必须是数字
            # 根据输入执行相应操作
            if num == 0:  # 用户选择退出
                cyan_print("Exiting...")
                break
            elif num == 1:
                cyan_print("Option 1 selected: Embedding watermark into video...")
                Watermark.embed_watermark_in_video(input_video, key=key, alpha=alpha)
                # 这里可以加入相应的功能代码
            elif num == 2:
                cyan_print("Option 2 selected: Comparing original video with watermarked video...")
                base, ext = os.path.splitext(input_video)
                watermarked_video = f"{base}_watermarked{ext}"
                # Detect the watermark in the watermarked video
                Watermark.detect_watermark_in_video(input_video, watermarked_video, key=key, alpha=alpha, threshold=0.0)
                # 这里可以加入相应的功能代码
            elif num == 3:
                cyan_print("Option 3 selected: Comparing original video with duplicated video...")

            elif num == 4:
                cyan_print("Option 4 selected: Extracting watermark...")
                base, ext = os.path.splitext(input_video)
                watermarked_video = f"{base}_watermarked{ext}"
                Watermark.extract_and_save_watermark(input_video, watermarked_video, key=key, alpha=alpha)
            else:
                cyan_print("Invalid option. Please enter a valid number.")

        except ValueError:
            cyan_print("Invalid input. Please enter a valid number.")

