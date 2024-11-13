import platform
import click
import functools
from halo import Halo
def detect_os():
    system = platform.system()
    if system == "Windows":
        return "Windows"
    elif system == "Darwin":
        return "macOS"
    else:
        return "Other"

def isWindows():
    return detect_os() == "Windows"

def isMac():
    return detect_os() == "Darwin"

def cyan_print(str):
    click.echo(click.style(str, fg="cyan"))

def blue_print(str):
    click.echo(click.style(str, fg="blue"))

def info_print(str):
    click.echo(click.style(str, fg="green"))


def tech_loading(func=None, *, text="⚡ PROCESSING\n"):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            spinner = Halo(text=text, spinner='dots', color='cyan', text_color='cyan')
            with spinner:
                result = func(*args, **kwargs)
            spinner.succeed('✓ COMPLETE\n')
            return result
        return wrapper
    return decorator if func is None else decorator(func)