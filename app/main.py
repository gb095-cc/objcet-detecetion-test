import sys
from PyQt6.QtWidgets import QApplication

import MainWinodw as mw

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = mw.MainWindow()
    window.show()
    sys.exit(app.exec())