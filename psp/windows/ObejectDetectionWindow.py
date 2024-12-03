from torchvision import models
import cv2
import numpy as np
from lime import lime_image
from PyQt6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QVBoxLayout,QHBoxLayout
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QTimer
from ultralytics import YOLO

class ObejectDetectionWindow(QWidget):
    def __init__(self,parent =None):
        super().__init__()
        self.model = YOLO('yolov5s.pt')  # 加载预训练的 YOLOv5 模型
        self.model.eval()
        self.parent = parent
        self.initUI()

    def initUI(self):
        self.setWindowTitle('视频目标检测')
        self.setGeometry(100, 100, 800, 600)
        
        self.layout = QVBoxLayout()
        self.video_label = QLabel('选择视频进行目标检测', self)
        self.layout.addWidget(self.video_label)

        self.button = QPushButton('选择视频', self)
        self.button.clicked.connect(self.upload_video)
        self.layout.addWidget(self.button)

        # 存储危险帧的标签
        self.danger_frames_layout = QHBoxLayout()
        self.danger_frames_labels = [QLabel(self) for _ in range(10)]
        for label in self.danger_frames_labels:
            label.setFixedSize(100, 100)
            self.danger_frames_layout.addWidget(label)
        self.layout.addLayout(self.danger_frames_layout)

        self.setLayout(self.layout)

        # 初始化模型
        self.model = models.resnet18(pretrained=True)
        self.model.eval()

        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)
        self.video_capture = None
        self.frame_count = 0

    def upload_video(self):
        fname, _ = QFileDialog.getOpenFileName(self, '选择视频', '', 'Videos (*.mp4 *.avi *.mov)')
        if fname:
            self.video_capture = cv2.VideoCapture(fname)
            self.frame_count = 0
            self.timer.start(30)  # 每30毫秒处理一帧

    def process_frame(self):
        if self.video_capture is not None:
            ret, frame = self.video_capture.read()
            if not ret:
                self.timer.stop()
                self.video_capture.release()
                return
            
            # 对帧进行目标检测（简化，实际需要模型逻辑）
            is_danger = self.detect_objects(frame)
            
            # 显示当前帧
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_image))

            # 如果是危险帧，保存并显示
            if is_danger:
                self.save_danger_frame(frame)

            self.frame_count += 1

    def detect_objects(self, frame):
        # 简化的检测逻辑，这里返回随机结果
        return np.random.rand() > 0.7  # 假设70%的概率判断为危险

    def save_danger_frame(self, frame):
        # 将危险帧转换为 QPixmap 并显示
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # 依次覆盖显示危险帧
        for i in range(9, 0, -1):
            self.danger_frames_labels[i].setPixmap(self.danger_frames_labels[i-1].pixmap())
        self.danger_frames_labels[0].setPixmap(QPixmap.fromImage(q_image))
        """Converts a PIL image to QPixmap"""
        image = image.convert("RGB")
        image_array = np.array(image)
        h, w, ch = image_array.shape
        bytes_per_line = ch * w
        q_image = QImage(image_array.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(q_image)
    
    
    def closeEvent(self, event):
        """当子窗口关闭时，显示主窗口"""
        self.parent.show()  # 重新显示主窗口
        event.accept()  # 接受关闭事件，允许窗口关闭

