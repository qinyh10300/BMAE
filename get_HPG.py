import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('datasets/cifar10_dataset/train/automobile/automobile_train_39.png', cv2.IMREAD_GRAYSCALE)

# 检查图像是否加载成功
if image is None:
    raise ValueError("Error: Image not found or failed to load.")

# 调整图像大小为 64x128（HOG 描述符的标准尺寸）
image = cv2.resize(image, (64, 128))

# 计算图像梯度（使用 Sobel 算子）
grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # x 方向梯度
grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # y 方向梯度

# 计算梯度幅值和方向
magnitude = cv2.magnitude(grad_x, grad_y)  # 梯度幅值
angle = cv2.phase(grad_x, grad_y, angleInDegrees=True)  # 梯度方向（角度）

# 可视化原始图像和梯度
plt.figure(figsize=(12, 8))

# 原始图像
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# x 方向梯度
plt.subplot(2, 2, 2)
plt.imshow(np.abs(grad_x), cmap='gray')
plt.title('Gradient X (Sobel)')
plt.axis('off')

# y 方向梯度
plt.subplot(2, 2, 3)
plt.imshow(np.abs(grad_y), cmap='gray')
plt.title('Gradient Y (Sobel)')
plt.axis('off')

# 梯度幅值
plt.subplot(2, 2, 4)
plt.imshow(magnitude, cmap='gray')
plt.title('Gradient Magnitude')
plt.axis('off')

plt.tight_layout()
plt.show()