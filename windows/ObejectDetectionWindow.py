from transformers import DetrForObjectDetection, DetrImageProcessor
import numpy as np
from PyQt6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QThread, pyqtSignal, QTimer
import torch
from PIL import Image, ImageDraw
import imageio


class VideoProcessingThread(QThread):
    frame_processed = pyqtSignal(np.ndarray)  # 信号，用于传递处理后的帧

    def __init__(self, video_path, processor, model, device):
        super().__init__()
        self.video_path = video_path
        self.processor = processor
        self.model = model
        self.device = device
        self.running = True
        self.video_reader = imageio.get_reader(video_path)
        self.frame_skip = 5  # 每隔 2 帧处理一次

    def run(self):
        for i, frame in enumerate(self.video_reader):
            if not self.running:
                break
            # 跳过未选中的帧
            if i % self.frame_skip != 0:
                continue
            
            # 处理帧
            processed_frame = self.detect_objects(frame)
            self.frame_processed.emit(processed_frame)  # 发送处理后的帧

    def detect_objects(self, frame):
        # 将帧转换为 PIL 图像并调整大小
        pil_image = Image.fromarray(frame)
        # pil_image = pil_image.resize((1200, 600), Image.Resampling.LANCZOS)

        # 预处理帧
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

        # 模型预测
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 获取检测结果
        logits = outputs.logits[0]  # (N, num_classes)
        boxes = outputs.pred_boxes[0]  # (N, 4)

        # 筛选检测到的物体
        scores = logits.softmax(-1)[:, :-1].max(-1).values  # 排除背景
        detected_indices = scores > 0.7  # 置信度阈值
        detected_boxes = boxes[detected_indices]
        detected_scores = scores[detected_indices]
        detected_classes = logits[detected_indices].argmax(-1)  # 获取类别索引

        # 类别索引映射到名称
        id2label = self.model.config.id2label

        # 使用 PIL 绘制检测框
        draw = ImageDraw.Draw(pil_image)
        for box, score, cls_idx in zip(detected_boxes, detected_scores, detected_classes):
            # 计算实际坐标
            x_min, y_min, x_max, y_max = (box * torch.tensor([800, 600, 800, 600])).int().tolist()

            # 确保坐标正确
            x_min, x_max = sorted([x_min, x_max])
            y_min, y_max = sorted([y_min, y_max])

            # 获取类别名称
            label = id2label[cls_idx.item()]

            # 绘制矩形框
            draw.rectangle([x_min, y_min, x_max, y_max], outline="green", width=2)

            # 绘制类别名称和置信度
            draw.text((x_min, max(0, y_min - 10)), f"{label} {score:.2f}", fill="green")

        return np.array(pil_image)

    def stop(self):
        self.running = False
        self.quit()
        self.wait()


class ObjectDetectionWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(self.device).eval()
        self.parent = parent
        self.initUI()

    def initUI(self):
        self.setWindowTitle('实时目标检测')
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()
        self.video_label = QLabel('选择视频进行目标检测', self)
        self.layout.addWidget(self.video_label)

        self.button = QPushButton('选择视频', self)
        self.button.clicked.connect(self.upload_video)
        self.layout.addWidget(self.button)

        self.setLayout(self.layout)

        self.video_path = None
        self.video_thread = None

    def upload_video(self):
        fname, _ = QFileDialog.getOpenFileName(self, '选择视频', '', 'Videos (*.mp4 *.avi *.mov)')
        if fname:
            self.video_path = fname

            # 创建并启动视频处理线程
            self.video_thread = VideoProcessingThread(self.video_path, self.processor, self.model, self.device)
            self.video_thread.frame_processed.connect(self.update_frame)
            self.video_thread.start()

    def update_frame(self, frame):
        # 更新视频帧到 UI
        rgb_image = frame  # frame 已经是 RGB 格式的 NumPy 数组
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def closeEvent(self, event):
        """当窗口关闭时，停止线程并释放资源"""
        if self.video_thread:
            self.video_thread.stop()
        event.accept()
