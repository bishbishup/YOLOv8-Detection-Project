import cv2
from ultralytics import YOLO
from faker import Faker

# 加载v8模型
model = YOLO('../model/yolov8l-face.pt')

# 创建一个Faker对象用于生成随机姓名
faker = Faker()

# 打开视频文件
video_path = "../media/00009.MTS"
cap = cv2.VideoCapture(video_path)

# 循环播放视频帧
while cap.isOpened():
    # 从视频中读取帧
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model.track(frame,persist=True)
        # for r in results:
        #     print(r.boxes)

        # 获取检测到的人脸列表
        faces = results[0].boxes.xyxy

        track_ids = results[0].boxes.id

        # Visualize the results on the `frame
        # Visualize the results on the `frame
        annotated_frame = frame.copy()
        for face in faces:
            x1, y1, x2, y2 = face
            # 绘制框选的人脸区域
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            # 生成随机姓名
            name = faker.name()
            # 将姓名写在人脸上方
            cv2.putText(annotated_frame, name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the annotated frame
        cv2.imshow("YOLOv5 Inference", annotated_frame)

        # 按ESC退出
        if cv2.waitKey(30) & cv2.waitKey(1) == 27:
            break
    else:
        # 视频放完退出
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()