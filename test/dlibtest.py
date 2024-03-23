import cv2
import dlib
from faker import Faker

# 读取视频
cap = cv2.VideoCapture('../media/00009.MTS')

# 初始化faker包
fake = Faker()

# 初始化已知人脸和对应的名字的字典
known_faces = {}

# 初始化人脸检测器和特征点预测器
detector = dlib.get_frontal_face_detector()
# 在官网下载训练数据集
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat")  # 替换为特征点预测器的路径

# 初始化已出现的人脸 ID 列表
face_ids = []

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # 转换为灰度图以进行人脸检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 人脸检测
        faces = detector(gray)

        # 遍历检测到的人脸
        for face in faces:
            # 提取人脸边界框位置
            (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())

            # 将 dlib.rectangle 类型转换为元组类型
            face_tuple = (x, y, w, h)

            if face_tuple not in known_faces:
                known_faces[face_tuple] = fake.name()

            # 在人脸框上标记名字
            cv2.putText(frame, known_faces[face_tuple], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
                        cv2.LINE_AA)

            # 在视频帧上绘制红色方框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # 显示处理后的视频帧
        cv2.imshow('Output Video', frame)

        # 设置适当的等待时间
        if cv2.waitKey(30) & cv2.waitKey(1) == 27:
            break
    else:
        break

# 释放资源并关闭输出视频文件
cap.release()
cv2.destroyAllWindows()