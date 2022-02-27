"""
 add color text tools for show progress and output more precisely
"""

class Colors:
    pref = "\033["
    reset = f"{pref}0m"
    black = "30m"
    red = "31m"
    green = "32m"
    yellow = "33m"
    blue = "34m"
    magenta = "35m"
    cyan = "36m"
    white = "37m"


### TEST
def test_ansi_black(text, color=Colors.black, is_bold=False):
    print(f'{Colors.pref}{1 if is_bold else 0};{color}' + text + Colors.reset)

def test_ansi_black_with_bold(text, color=Colors.black, is_bold=True):
    print(f'{Colors.pref}{1 if is_bold else 0};{color}' + text + Colors.reset)

def test_ansi_red(text, color=Colors.red, is_bold=False):
    print(f'{Colors.pref}{1 if is_bold else 0};{color}' + text + Colors.reset)

def test_ansi_red_bold(text, color=Colors.red, is_bold=True):
    print(f'{Colors.pref}{1 if is_bold else 0};{color}' + text + Colors.reset)

def test_ansi_green(text, color=Colors.green, is_bold=False):
    print(f'{Colors.pref}{1 if is_bold else 0};{color}' + text + Colors.reset)

def test_ansi_green_with_bold(text, color=Colors.green, is_bold=True):
    print(f'{Colors.pref}{1 if is_bold else 0};{color}' + text + Colors.reset)

def test_ansi_yellow(text, color=Colors.yellow, is_bold=False):
    print(f'{Colors.pref}{1 if is_bold else 0};{color}' + text + Colors.reset)

def test_ansi_yellow_with_bold(text, color=Colors.yellow, is_bold=False):
    print(f'{Colors.pref}{1 if is_bold else 0};{color}' + text + Colors.reset)

def test_ansi_blue(text, color=Colors.blue, is_bold=False):
    print(f'{Colors.pref}{1 if is_bold else 0};{color}' + text + Colors.reset)

def test_ansi_blue_with_bold(text, color=Colors.blue, is_bold=True):
    print(f'{Colors.pref}{1 if is_bold else 0};{color}' + text + Colors.reset)

def test_ansi_magenta(text, color=Colors.magenta, is_bold=False):
    print(f'{Colors.pref}{1 if is_bold else 0};{color}' + text + Colors.reset)

def test_ansi_magenta_with_bold(text, color=Colors.magenta, is_bold=True):
    print(f'{Colors.pref}{1 if is_bold else 0};{color}' + text + Colors.reset)

def test_ansi_cyan(text, color=Colors.cyan, is_bold=False):
    print(f'{Colors.pref}{1 if is_bold else 0};{color}' + text + Colors.reset)

def test_ansi_cyan_with_bold(text, color=Colors.cyan, is_bold=True):
    print(f'{Colors.pref}{1 if is_bold else 0};{color}' + text + Colors.reset)

def test_ansi_white(text, color=Colors.white, is_bold=False):
    print(f'{Colors.pref}{1 if is_bold else 0};{color}' + text + Colors.reset)

def test_ansi_white_with_bold(text, color=Colors.white, is_bold=True):
    print(f'{Colors.pref}{1 if is_bold else 0};{color}' + text + Colors.reset)




if __name__ == '__main__':
    test_ansi_black('hello')
    test_ansi_black_with_bold('hello')

    test_ansi_blue('hello')
    test_ansi_blue_with_bold('hello')

    test_ansi_cyan('hello')
    test_ansi_cyan_with_bold('hello')

    test_ansi_green('hello')
    test_ansi_green_with_bold('hello')

    test_ansi_magenta('hello')
    test_ansi_green_with_bold('hello')

    test_ansi_red('hello')
    test_ansi_red_bold('hello')

    test_ansi_white('hello')
    test_ansi_white_with_bold('hello')
