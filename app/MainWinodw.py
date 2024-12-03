from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QVBoxLayout, QWidget, QPushButton
from PyQt6.QtGui import QColor, QPalette
from windows import ImageClasscificationWindow as icw
from windows import ObejectDetectionWindow as odw

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()      
        self.initUI()

    def initUI(self):
        # 设置窗口标题
        self.setWindowTitle("深度学习使用工具")
        self.setGeometry(100, 100, 800, 600)

        # 设置窗口背景颜色为蓝色
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor('white'))
        self.setPalette(palette)

        # 创建三个按钮
        self.button1 = QPushButton('图像分类任务')
        self.button2 = QPushButton('目标检测任务')
        self.button3 = QPushButton('后门实验任务')

        # 创建布局并添加按钮
        layout = QVBoxLayout()
        layout.addWidget(self.button1)
        layout.addWidget(self.button2)
        layout.addWidget(self.button3)

        # 设置窗口的布局
        self.setLayout(layout)

        self.buttonConnect()

    def buttonConnect(self):
        self.button1.clicked.connect(self.openicw)
        self.button2.clicked.connect(self.openodb)
        # self.button3.clicked.connect()

    def openicw(self):
        self.hide()
        self.window = icw.ImageClassifierWindow(self)
        self.window.show()

    def openodb(self):
        self.hide()
        self.window = odw.ObjectDetectionWindow(self)
        self.window.show()


