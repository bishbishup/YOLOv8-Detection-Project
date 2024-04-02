import cv2
from ultralytics import YOLO
from faker import Faker
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import dlib

# 人脸关键点检测器
predictor_path = "model/shape_predictor_68_face_landmarks.dat"
# 人脸识别模型、提取特征值
face_rec_model_path = "model/dlib_face_recognition_resnet_model_v1.dat"
# 加载模型
# 人脸检测
detector = dlib.get_frontal_face_detector()
# 关键点检测
sp = dlib.shape_predictor(predictor_path)
# 编码
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# 实例化YOLO模型并将其移动到设备上
model = YOLO('model/yolov8l-face.pt')

# 创建一个Faker对象用于生成随机姓名sms
faker = Faker(locale='zh_CN')

# 打开视频文件并获取视频参数
video_path = "media/00009.MTS"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# 设置输出视频参数
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('result/Student.mp4', fourcc, fps, size, True)


# 判断人脸照片是否有特征
def judge_character(image):
    # 人脸检测
    dets = detector(image, 1)
    if len(dets) == 1:
        # 关键点
        shape = sp(image, dets[0])
        # 提取特征
        # 获取到128位的编码
        face_descriptor = facerec.compute_face_descriptor(image, shape)
        v = np.array(face_descriptor)
        return v
    else:
        return None


# cv2将标识结果可视化
def box_label(origin, studentId, image, b, label='', txt_color=(255, 255, 255)):
    # 得到目标矩形框的左上角和右下角坐标(xx,yy)
    p1, p2 = (int(b[0]), int(b[1])), (int(b[2]), int(b[3]))
    # 获取可以识别出的人脸的文件夹中的所有文件名
    rec_file_names = os.listdir('rec-face')
    # 将每个人的人脸单独提取出来
    face = origin[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
    face2 = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    result = judge_character(face2)
    # 如果识别出来则记录进识别列表
    if result is not None and track_id_number not in rec_list:
        rec_list.append(track_id_number)
    # 如果检测出人脸则使用绿色框框出来
    # 因为后续该人脸截取到的图片不一定能提取到人脸特征，所以识别出来后面可能不会显示绿色
    # 将识别出来的id记录下来，这样子后续即使识别不出人脸特征也能直接给其框出绿色方框
    if result is not None or track_id_number in rec_list:
        # 如果文件夹中识别出该人脸则存入进文件夹
        if f'{int(studentId)}-{label}.png' not in rec_file_names:
            cv2.imwrite(os.path.join('rec-face', f'{int(studentId)}-{label}.png'), face)
        # 绘制绿色矩形框
        cv2.rectangle(image, p1, p2, (0,255,0), thickness=1, lineType=cv2.LINE_AA)
        # 得到要书写的文本的宽和高，用于给文本绘制背景色
        # 由于我的名字是中文名，所以getTextSize获取到其w不准确,h还是可以的
        w, h = cv2.getTextSize(label, 0, fontScale=2 / 3, thickness=1)[0]
        # # 打算将名字划分成两字、三字和四字的
        if len(label) == 2:
            w = 40
        elif len(label) == 3:
            w = 60
        elif len(label) == 4:
            w = 80
        # 确保显示的文本不会超出图片范围
        outside = p1[1] - h >= 3
        # p1[1] -> y
        p3 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        # 给文字填充背景颜色
        cv2.rectangle(image, p1, p3, (0, 255, 0), -1, cv2.LINE_AA)
        # 解决cv2.putText中文输出？的方法
        # 设置字体大小
        textSize = 20
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(img)
        # 字体的格式
        fontStyle = ImageFont.truetype(
            "SimHei.ttf", textSize, encoding="utf-8")
        # 绘制文本
        draw.text((p1[0], p3[1] - 2 if outside else p1[1] + h + 2), label, (0,0,0), font=fontStyle)
        # 转换回OpenCV格式
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    else:
        # 绘制红色矩形框
        cv2.rectangle(image, p1, p2, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
        # 得到要书写的文本的宽和高，用于给文本绘制背景色
        # 由于我的名字是中文名，所以getTextSize获取到其w不准确,h还是可以的
        w, h = cv2.getTextSize(label, 0, fontScale=2 / 3, thickness=1)[0]
        # 这里只有一种情况就是无法识别，所以w为80
        w = 80
        # 确保显示的文本不会超出图片范围
        outside = p1[1] - h >= 3
        # p1[1] -> y
        p3 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        # 给文字填充背景颜色
        cv2.rectangle(image, p1, p3, (0, 0, 255), -1, cv2.LINE_AA)  # 填充颜色
        # 解决cv2.putText中文输出？的方法
        # 设置字体大小
        textSize = 20
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(img)
        # 字体的格式
        fontStyle = ImageFont.truetype(
            "SimHei.ttf", textSize, encoding="utf-8")
        # 绘制文本
        draw.text((p1[0], p3[1] - 2 if outside else p1[1] + h + 2), '无法识别', txt_color, font=fontStyle)
        # 转换回OpenCV格式
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


# 存储id对应的名字的字典，
nameBox = {}
# 存储已识别出人脸的id的列表
rec_list = []

# 循环播放视频帧
while cap.isOpened():
    # 从视频中读取帧
    success, frame = cap.read()

    if success:

        # Run YOLOv8 inference on the frame
        results = model.track(frame, conf=0.3, persist=True)
        # for r in results:
        #     print(r.boxes)

        track_ids = []
        # 得到该帧的各个目标的ID
        if results[0].boxes.id is not None and len(results) > 0:
            # print(results[0].boxes.id)
            for element in results[0].boxes.id:
                # 如果id没在字典里则调用faker方法
                if element.item() not in nameBox.keys():
                    # print(element.item())
                    nameBox[element.item()] = faker.name()  # 给字典添加键值对
                    # track_ids = results[0].boxes.id.int().cpu().tolist()
                    temp = [element.item(), nameBox[element.item()]]
                    track_ids.append(temp)
                # 如果在的话就使用字典里已经存储好的名字
                else:
                    temp = [element.item(), nameBox[element.item()]]
                    track_ids.append(temp)

        # 复制原图像
        frame_copy = frame.copy()
        # 遍历该帧的所有目标
        for track_id, box in zip(track_ids, results[0].boxes.data):
            track_id_number, person_name = track_id
            frame = box_label(frame_copy, track_id_number, frame, box, str(person_name), (255, 255, 255))

        # 显示出识别到的人脸数量
        cv2.putText(frame, 'Student: ' + str(int(len(rec_list))), (int(0.35 * size[0]), int(0.1 * size[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

        # 显示标记好的当前帧图像
        cv2.imshow("识别视频", frame)

        # 将帧写入视频
        out.write(frame)

        # 按ESC退出
        if cv2.waitKey(30) & cv2.waitKey(1) == 27:
            break
    else:
        # 视频放完退出
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
