# 测试result的plot方法
from PIL import Image
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('../model/yolov8n.pt')

# Run inference on
results = model('car2.jpg')  # results list

# Visualize the results
for r in results:
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()

    # Save results to disk
    r.save(filename='../media/results.jpg')