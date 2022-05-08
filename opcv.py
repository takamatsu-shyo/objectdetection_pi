import cv2
import numpy as np
from queue import Queue
from threading import Thread
import time


def capture(img_que):
    cap = cv2.VideoCapture("data/test.mp4")
    while(1):
        ret, img = cap.read()
        if(ret):
            img_que.put(img)

def inference(img_que):

    Width = 1280 
    Height = 720

    net = cv2.dnn.readNet("data/yolov3-tiny.weights", "data/yolov3-tiny.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    scale = 0.00392
    conf_threshold = 0.5
    nms_threshold = 0.4
    frame_counter = 0
    start_time = time.time()

    while(1):
        frame_counter += 1
        if(frame_counter > 27):
            print("EOF")
            import os
            os._exit(1)

        class_ids = []
        confidences = []
        boxes = []

        img = img_que.get()

        blob = cv2.dnn.blobFromImage(
            img, scale, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, conf_threshold, nms_threshold)
        print(frame_counter * 1.0 / (time.time() - start_time))


if __name__ == "__main__":
    img_que = Queue(maxsize=1)
    Thread(target=capture, args=(img_que,)).start()
    Thread(target=inference, args=(img_que,)).start()
