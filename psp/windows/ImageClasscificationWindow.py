import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
from lime import lime_image
from PyQt6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QTextEdit
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtCore import Qt
import Tool.GetFileData as gfd


index2label = gfd.GetJson()

class ImageClassifierWindow(QWidget):
    def __init__(self,parent =None):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.eval()
        self.initUI()
        self.parent = parent

    def initUI(self):
        self.setWindowTitle('图像分类检测')
        self.setGeometry(100, 100, 800, 600)  # Set a bigger window size
        
        # 布局
        self.layout = QVBoxLayout()
        
        # 标题
        self.label = QLabel('图像分类与解释器', self)
        self.label.setFont(QFont('Arial', 24))
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.label)
        
        # 上传图像 QLabel
        self.origin_label = QLabel(self)
        self.origin_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.origin_label)

        # 解释图像 QLabel
        self.explanation_label = QLabel(self)
        self.explanation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.explanation_label)
        
        # 文本输出
        self.result_text = QTextEdit(self)
        self.result_text.setReadOnly(True)
        self.result_text.setFont(QFont('Arial', 14))  # Set a readable font for results
        self.result_text.setStyleSheet("padding: 10px; background-color: #f0f0f0;")  # Styling for readability
        self.layout.addWidget(self.result_text)

        # 按钮
        self.button = QPushButton('选择图片', self)
        self.button.setFont(QFont('Arial', 16))  # Larger font for button
        self.button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")  # Button styling
        self.button.clicked.connect(self.upload_image)
        self.layout.addWidget(self.button)
        
        
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.setLayout(self.layout)

    def upload_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, '选择图片', '', 'Images (*.png *.jpg *.jpeg)')
        if fname:
            image = Image.open(fname).convert("RGB")
            predictionlabel, explanation_image = self.classify_and_explain(image)

            
            q_image = self.pil2pixmap(image)
            self.origin_label.setPixmap(q_image)

            
            self.label.setText(f'分类结果: {predictionlabel}')
            self.result_text.setPlainText(f'解释图像生成成功!')

            
            self.display_explanation_image(explanation_image)
    #图像预处理
    def preprocess_image(self, image):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return preprocess(image).unsqueeze(0)

    def classify_and_explain(self, image):

        # 预测
        image_tensor = self.preprocess_image(image)
        with torch.no_grad():
            output = self.model(image_tensor)
        _, predicted_idx = torch.max(output, 1)
        
        # LIME 解释
        image_array = np.array(image).astype(np.float32) / 255
        if image_array.ndim == 2:  # 如果是灰度图像
            image_array = np.stack((image_array,) * 3, axis=-1)  # 转换为 RGB

        def model_predict(x):
            x_tensor = torch.from_numpy(x).permute(0, 3, 1, 2)  # 从 (H, W, C) 转换为 (N, C, H, W)
            return self.model(x_tensor).detach().numpy()

        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            image_array,
            model_predict,
            top_labels=5,
            hide_color=0,
            num_samples=1000
        )

        # 使用 explanation 中的 mask 生成解释图像
        label = explanation.top_labels[0]  # 获取最重要的标签
        temp, mask = explanation.get_image_and_mask(label, positive_only=True, num_features=10, hide_rest=False)

        # 创建解释图像
        explanation_image = (temp * 255).astype(np.uint8)

        # 调整解释图像的颜色以突出显示
        explanation_image[mask == 0] = [0, 0, 0]  # 将不重要区域设为黑色

        return index2label[predicted_idx.item()], explanation_image  # 返回预测和解释图像

    def display_explanation_image(self, explanation_image):
        # Ensure the explanation image is in RGB format
        if explanation_image.shape[2] == 4:  # 如果是 RGBA，去掉 alpha 通道
            explanation_image = explanation_image[:, :, :3]

        # Convert explanation image to QImage
        h, w, ch = explanation_image.shape
        q_image = QImage(explanation_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        # Display in QLabel
        self.explanation_label.setPixmap(pixmap)

    def pil2pixmap(self, image):
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

