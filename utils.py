import platform
import click
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