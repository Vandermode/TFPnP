import os


class COLOR:
    DEFAULT = '\033[00m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    LIGHT_PURPLE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    LIGHT_GRAY = '\033[97m'
    BLACK = '\033[98m'


def xprint(*args, color=COLOR.DEFAULT, **kwargs):
    print(color, end='')
    print(*args, **kwargs)
    print(COLOR.DEFAULT, end='')


class Logger:
    def __init__(self, log_dir=None):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, 'log.txt') if log_dir else None
        if self.log_path is not None:
            self.log_file = open(self.log_path, 'a')

    def log(self, *args, color=COLOR.DEFAULT):
        xprint(*args, color=color)

        if self.log_path is not None:
            with open(self.log_path, 'a') as f:
                content = ' '.join(list(map(str, args))) + '\n'
                f.write(content)


if __name__ == '__main__':
    # xprint('hello', 'asdf', color=RED, end='xxx')
    logger = Logger('./')
    logger.log('hello', 'asdf', color=COLOR.RED)
