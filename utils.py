import platform

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