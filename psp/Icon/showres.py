import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取干净图像和扰动图像
clean_image = cv2.imread(r'D:\上课材料\高级软件工程\code\app\Icon\dog.png')
perturbed_image = cv2.imread(r'D:\上课材料\高级软件工程\code\app\Icon\image_with_trigger.png')

# 转换为灰度图（可选，视图像类型而定）
clean_image_gray = cv2.cvtColor(clean_image, cv2.COLOR_BGR2GRAY)
perturbed_image_gray = cv2.cvtColor(perturbed_image, cv2.COLOR_BGR2GRAY)

# 计算残差（绝对差异）
residuals = np.abs(clean_image_gray.astype(np.float32) - perturbed_image_gray.astype(np.float32))

# 放大残差以便于可视化（选择适当的放大因子，视实际残差大小而定）
residuals_enhanced = residuals * 100  # 放大100倍

# 将残差归一化到[0, 255]范围以适应显示
residuals_enhanced = np.clip(residuals_enhanced, 0, 255).astype(np.uint8)

# 显示残差图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(clean_image_gray, cmap='gray')
plt.title('Clean Image')

plt.subplot(1, 3, 2)
plt.imshow(perturbed_image_gray, cmap='gray')
plt.title('Perturbed Image')

plt.subplot(1, 3, 3)
plt.imshow(residuals_enhanced, cmap='hot')
plt.title('Residuals')

plt.show()
