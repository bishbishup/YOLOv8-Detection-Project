# 提取人脸特征
import cv2
from mtcnn import MTCNN
import numpy as np

# 初始化 MTCNN 模型
detector = MTCNN()

# 读取图像
image = cv2.imdecode(np.fromfile('face/3-徐莉.png', dtype=np.uint8),-1)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 检测人脸
faces = detector.detect_faces(rgb_image)
for face in faces:
    bbox = face['box']
    keypoints = face['keypoints']

    # 绘制人脸框和特征点
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)
    cv2.circle(image, (keypoints['left_eye']), 2, (0, 255, 0), 2)
    cv2.circle(image, (keypoints['right_eye']), 2, (0, 255, 0), 2)
    cv2.circle(image, (keypoints['nose']), 2, (0, 255, 0), 2)
    cv2.circle(image, (keypoints['mouth_left']), 2, (0, 255, 0), 2)
    cv2.circle(image, (keypoints['mouth_right']), 2, (0, 255, 0), 2)

# 显示标记后的图像
cv2.imshow("Facial Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()