import cv2
import numpy as np

# 读取图像
image = cv2.imdecode(np.fromfile('face/2-刘秀梅.png', dtype=np.uint8),-1)

# 计算图像的清晰度
sharpness = cv2.Laplacian(image, cv2.CV_64F).var()

print(f"图像清晰度为: {sharpness}")