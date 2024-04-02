import cv2
from ultralytics import YOLO
from collections import defaultdict

model = YOLO('model/yolov8l.pt')

cap = cv2.VideoCapture("media/video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fNUMS = cap.get(cv2.CAP_PROP_FRAME_COUNT)

# 设置输出视频参数
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('result/car.mp4', fourcc, fps, size)


def box_label(image, boxs, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    # 得到目标矩形框的左上角和右下角坐标
    p1, p2 = (int(boxs[0]), int(boxs[1])), (int(boxs[2]), int(boxs[3]))
    # 绘制矩形框
    cv2.rectangle(image, p1, p2, color, thickness=1, lineType=cv2.LINE_AA)
    if label:
        # 得到要书写的文本的宽和长，用于给文本绘制背景色
        w, h = cv2.getTextSize(label, 0, fontScale=2 / 3, thickness=1)[0]
        # 确保显示的文本不会超出图片范围
        p2 = p1[0] + w, p1[1] - h - 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # 填充颜色
        # 书写文本
        cv2.putText(image,
                    label, (p1[0], p1[1] - 2),
                    0,
                    2 / 3,
                    txt_color,
                    thickness=1,
                    lineType=cv2.LINE_AA)


# track_history用于保存目标ID，以及它在各帧的目标位置坐标，这些坐标是按先后顺序存储的
track_history = defaultdict(lambda: [])
# 车辆的计数变量
# 这里视频中的前辆车是不在后面的判断标准范围内的，所以先给这俩加一
car_right = 0
car_left = 0

# 视频帧循环
while cap.isOpened():
    # 读取一帧图像
    success, frame = cap.read()

    if success:
        # 在帧上运行YOLOv8跟踪，persist为True表示保留跟踪信息，conf为0.3表示只检测置信值大于0.3的目标
        # 规定对于该帧数下面区域的车辆进行识别
        results = model.track(frame[300:720,0:1280], conf=0.3, persist=True)

        track_ids = []
        # 得到该帧的各个目标的ID
        if results[0].boxes.id is not None and len(results) > 0:
            for element in results[0].boxes.id:
                track_ids.append(int(element.item()))

        # 遍历该帧的所有目标
        for track_id, box in zip(track_ids, results[0].boxes.data):
            # 当车辆走进一定范围内再开始识别
            # 这里其实主要是为了消除割裂感
            if box[:4][1] + 300 > 350:
                # 绘制该目标的矩形框
                box_label(frame[300:720,0:1280], box, '#' + str(track_id) + ' car', (167, 146, 11))
            # 得到该目标矩形框的中心点坐标(x, y)
            # 注意，由于我上面使用yolo追踪的frame不是完整的，所以获取到的坐标要加上三百才是原来整张图的坐标
            x1, y1, x2, y2 = box[:4]
            x = (x1 + x2) / 2 + 300
            # y坐标也需要往下降
            y = (y1 + y2) / 2 + 300
            # 提取出该ID的以前所有帧的目标坐标，当该ID是第一次出现时，则创建该ID的字典
            track = track_history[track_id]
            track.append((float(x), float(y)))  # 追加当前目标ID的坐标
            # 只有当track中包括两帧以上的情况时，才能够比较前后坐标的先后位置关系
            if len(track) > 1:
                _, fore_y = track[-2]  # 提取前一帧的目标纵坐标
                # 我们设基准线为纵坐标是size[1]-100的水平线
                # 当前一帧在基准线的上面，当前帧在基准线的下面时，说明该车是从上往下运行
                if fore_y < size[1] - 150 <= y:
                    car_left += 1  # out计数加1
                # 当前一帧在基准线的下面，当前帧在基准线的上面时，说明该车是从下往上运行
                if fore_y > size[1] - 150 >= y:
                    car_right += 1  # in计数加1

            # # 目标为汽车的编号为2，其余的5为巴士，7为卡车，3为摩托车
            # if box[-1] == 2:
            # elif box[-1] == 5:  # 目标为巴士
            #     box_label(frame, box, '#' + str(track_id) + ' bus', (67, 161, 255))
            # elif box[-1] == 7:  # 目标为卡车
            #     box_label(frame, box, '#' + str(track_id) + ' truck', (19, 222, 24))
            # elif box[-1] == 3:  # 目标为摩托车
            #     box_label(frame, box, '#' + str(track_id) + ' motor', (186, 55, 2))

        # 绘制左边的基准线
        cv2.line(frame, (15, 570), (560, 570), color=(25, 33, 189), thickness=2,
                 lineType=4)
        # 绘制右边的基准线
        cv2.line(frame, (688, 570), (1159, 570), color=(25, 33, 189), thickness=2,
                 lineType=4)

        # 实时显示左边车辆统计信息
        cv2.putText(frame, str(car_left), (300, size[1] - 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # 实时显示右边车辆统计信息
        cv2.putText(frame, str(car_right), (950, size[1] - 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("YOLOv8 Tracking", frame)  # 显示标记好的当前帧图像
        out.write(frame)  # 写入保存

        # 按ESC退出
        if cv2.waitKey(30) & cv2.waitKey(1) == 27:
            break

    else:  # 视频播放结束时退出循环
        break

# 释放视频捕捉对象，并关闭显示窗口
cap.release()
out.release()
cv2.destroyAllWindows()
