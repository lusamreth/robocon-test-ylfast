import cv2 as cv
import torch

import model.detector
from time import time
import utils.utils
import numpy as np

# p = torch.load("../coco.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# cfg = utils.utils.load_datafile("./data/coco.data")
cfg = utils.utils.load_datafile("./fast-lagori.data")

model = model.detector.Detector(
    cfg["classes"], cfg["anchor_num"], True
).to(device)

# model.load_state_dict(torch.load("../coco.pth", map_location=device))
model.load_state_dict(
    torch.load("./fast-lagori-990.pth", map_location=device)
)

# # sets the module in eval node
model.eval()


def img_preparation(ori_img):
    # resize / scale down img to input spec
    res_img = cv.resize(
        ori_img,
        (cfg["width"], cfg["height"]),
        interpolation=cv.INTER_LINEAR,
    )

    img = res_img.reshape(1, cfg["height"], cfg["width"], 3)
    img = torch.from_numpy(img.transpose(0, 3, 1, 2))
    # forward to device
    img = img.to(device).float() / 255.0
    return img


LABEL_NAMES = ["lagori-disc", "banana"]


def detection_function(input_img):

    with torch.no_grad():
        data_img = img_preparation(input_img)
        preds = model(data_img)
        output = utils.utils.handel_preds(preds, cfg, device)
        output_boxes = utils.utils.non_max_suppression(
            output, conf_thres=0.3, iou_thres=0.4
        )

        h, w, _ = input_img.shape
        scale_h, scale_w = h / cfg["height"], w / cfg["width"]
        for box in output_boxes[0]:
            box = box.tolist()

            obj_score = box[4]
            category = LABEL_NAMES[int(box[5])]
            print(category)

            x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
            x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)

            cv.rectangle(
                input_img, (x1, y1), (x2, y2), (255, 255, 0), 2
            )
            cv.putText(
                input_img,
                "%.2f" % obj_score,
                (x1, y1 - 5),
                0,
                0.7,
                (0, 255, 0),
                2,
            )


cap = cv.VideoCapture("../lagori_ref_vid.mp4")

cao = cv.imread("./lagori-test.jpg")
detection_function(cao)


while True:
    cv.imshow("laa", cao)
    if cv.waitKey(25) & 0xFF == ord("q"):
        break


while True:
    start_time = time()
    _, frame = cap.read()

    detection_function(frame)
    end_time = time()
    fps = 1 / np.round(end_time - start_time, 3)
    print(f"Frames Per Second : {fps}")

    cv.putText(
        frame,
        str(fps),
        (7, 20),
        cv.FONT_HERSHEY_COMPLEX_SMALL,
        1,
        (255, 0, 255),
    )
    cv.imshow("fram", frame)
    if cv.waitKey(25) & 0xFF == ord("q"):
        break

# p("../green_yellow.jpeg")
