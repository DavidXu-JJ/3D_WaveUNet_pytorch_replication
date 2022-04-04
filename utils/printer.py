import os
from datetime import datetime

class Printer():
    def __init__(self, file):
        self.file = file
        self.open_or_close = False
        self._check()
        self._open()

    def _check(self):
        path, _ = os.path.split(self.file)
        assert os.path.isdir(path)

    def _open(self):
        self.info = open(self.file, 'w')
        self.open_or_close = True

    def _close(self):
        self.info.close()
        self.open_or_close = False

    def pprint(self, text):
        time = '[{}]'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + ' ' * 4
        print(time + text)
        self.info.write(time + text + '\n')


