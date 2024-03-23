import cv2
from ultralytics import YOLO
from faker import Faker
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# 实例化YOLO模型并将其移动到设备上
model = YOLO('model/yolov8l-face.pt')

# 创建一个Faker对象用于生成随机姓名
faker = Faker(locale='zh_CN')

# 打开视频文件并获取视频参数
video_path = "media/00009.MTS"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# 设置输出视频参数
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('result/result.mp4', fourcc, fps, (1920, 1080), True)


def box_label(origin, studentId, image, b, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    # 得到目标矩形框的左上角和右下角坐标(xx,yy)
    p1, p2 = (int(b[0]), int(b[1])), (int(b[2]), int(b[3]))
    # 绘制矩形框
    cv2.rectangle(image, p1, p2, color, thickness=1, lineType=cv2.LINE_AA)
    # 获取人脸照片的面积
    area = (int(b[2]) - int(b[0])) * (int(b[3]) - int(b[1]))
    # 获取人脸文件夹中的所有文件名
    file_names = os.listdir('face')
    # bug:重复的名字和id会被覆盖但是这样貌似就达成了去重的效果
    if area > 2500:
        # 将每个人的人脸单独提取出来
        face = origin[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
        # 如果文件夹中没有该人脸则存入进文件夹
        if f'{int(studentId)}-{label}.png' not in file_names:
            cv2.imwrite(os.path.join('face', f'{int(studentId)}-{label}.png'), face)
        # 如果有的话与文件夹中的文件比较大小，大的替换小的
        else:
            file_path = os.path.join('face',f'{int(studentId)}-{label}.png')
            with Image.open(file_path) as img:
                width, height = img.size
                if area > width * height:
                    cv2.imwrite(os.path.join('face', f'{int(studentId)}-{label}.png'), face)
    if label:
        # 得到要书写的文本的宽和高，用于给文本绘制背景色
        # 由于我的名字是中文名，所以getTextSize获取到其w不准确,h还是可以的
        w, h = cv2.getTextSize(label, 0, fontScale=2 / 3, thickness=1)[0]
        # print(w)
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
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, (0, 0, 0), -1, cv2.LINE_AA)  # 填充颜色
        # 书写文本
        # cv2.putText(image,
        #             label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
        #             0,
        #             2 / 3,
        #             txt_color,
        #             thickness=1,
        #             lineType=cv2.LINE_AA)
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
        draw.text((p1[0], p2[1] - 2 if outside else p1[1] + h + 2), label, txt_color, font=fontStyle)
        # 转换回OpenCV格式
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


# 存储id对应的名字的字典
nameBox = {}
# 存储最大id的数组
max_id_list = []

# 循环播放视频帧
while cap.isOpened():
    # 从视频中读取帧
    success, frame = cap.read()

    if success:

        # Run YOLOv8 inference on the frame
        results = model.track(frame, persist=True)
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

        # print(track_ids)

        # 临时存储该帧中id最大的一个数
        temp_id = []
        # 复制原图像
        frame_copy = frame.copy()
        # 遍历该帧的所有目标
        for track_id, box in zip(track_ids, results[0].boxes.data):
            track_id_number, person_name = track_id
            frame = box_label(frame_copy, track_id_number, frame, box, str(person_name), (0, 0, 255))
            temp_id.append(track_id_number)

        # 由于有些帧数最大的id被遮挡住了所以没有记录下来，所以需要一个存储最大id的数组
        if temp_id:
            max_id_list.append(max(temp_id))
        put_id = max(max_id_list)
        cv2.putText(frame, 'Count: ' + str(int(put_id)), (size[0] - 200, size[1] - 1000),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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
