import cv2
import numpy as np
from queue import Queue
from threading import Thread, Lock
import time


def capture_to_buffer():
    cap = cv2.VideoCapture("data/test.mp4")
    frame_buffer = []
    
    try:
        #while(1):
        for i in range(30):
            print(".", end="", flush=True)
            ret, img = cap.read()
            if(ret):
                frame_buffer.append(img)
            else:
                break
    except Exception as e:
        print(e)

    return frame_buffer


def main():

    frame_buffer = capture_to_buffer()
    print("frame buffer complete")

    Width = 1280 
    Height = 720

    #net = cv2.dnn.readNet("data/yolov3.weights", "data/yolov3.cfg")
    net = cv2.dnn.readNet("data/yolov3-tiny.weights", "data/yolov3-tiny.cfg")

    layer_names = net.getLayerNames()
    #output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    output_layers = net.getUnconnectedOutLayersNames()

    scale = 0.00392
    conf_threshold = 0.1
    nms_threshold = 0.4
    frame_counter = 0
    start_time = time.time()
    fps = 0.0

    for img in frame_buffer:
        frame_counter += 1

        class_ids = []
        confidences = []
        boxes = []

        blob = cv2.dnn.blobFromImage(img, scale, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        for out in outs:
            # out(507, 85), (2028, 85)
            for detection in out:
                # detection(85,)
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
                    print(class_ids, confidence,)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        fps = frame_counter * 1.0 / (time.time() - start_time)

        print(".", end="", flush=True)

    print(fps, " fps")


if __name__ == "__main__":
    main()
