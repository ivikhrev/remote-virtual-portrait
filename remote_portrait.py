import logging
import sys
import os
print(os.path.abspath(__file__))

from PyQt5.QtWidgets import QApplication
base_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, base_dir + '/src/ui')
from src.modules.config import Config
from src.ui.execution import Execution

log = logging.getLogger('Global log')
log_handler = logging.StreamHandler()
log.addHandler(log_handler)
log.setLevel(logging.DEBUG)
log_handler.setLevel(logging.DEBUG)
log_handler.setFormatter(logging.Formatter('[ %(levelname)s ] %(message)s'))


def main():
    config = Config(file_path=base_dir + "/config.json", project_path = base_dir)
    app = QApplication(sys.argv)
    ex = Execution(config)
    app.exec_()


if __name__ == '__main__':
    sys.exit(main() or 0)

