def colorstr(*input):
    *arg, string = input if len(input) > 1 else ("blue", "bold", input[0])
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "black": "\033[90m",
        "bold": "\033[1m",
        "end": "\033[0m",
        "reset": "\033[0m"
    }
    return "".join(colors[x] for x in arg) + f"{string}" + colors["end"]
