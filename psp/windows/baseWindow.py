from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit, QFileDialog
from PyQt6.QtGui import QColor, QPalette, QPixmap, QImage, QFont
from PyQt6.QtCore import Qt
from PIL import Image
import numpy as np
class BaseWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.initBaseUI()

    def initBaseUI(self):
        1


    def closeEvent(self, event):
        """关闭子窗口时重新显示父窗口"""
        if self.parent:
            self.parent.show()
        event.accept()
