import cv2
import numpy as np
import time
import queue
from threading import Thread, Lock

def capture(img_queue,cap,):
    try:
        while(True):
            ret, img = cap.read()
            img_queue.put(img)
    except Exception as e:
        print("Read frame ",e)


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    with open("data/coco.names", 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    label = str(classes[class_id])
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), (0, 128, 128), 5)
    cv2.putText(img, label, (x-20, y-20), cv2.FONT_HERSHEY_PLAIN, 5, (0, 128, 128), 2)


def inference(img_queue, w_h,):

    #net = cv2.dnn.readNet("data/yolov3.weights", "data/yolov3.cfg")
    net = cv2.dnn.readNet("data/yolov3-tiny.weights", "data/yolov3-tiny.cfg")

    output_layers = net.getUnconnectedOutLayersNames()

    scale = 0.00392
    conf_threshold = 0.1
    nms_threshold = 0.4
    frame_counter = 0
    fps = 0.0

    save_frame_buffer = []

    while(img_queue.empty()!=True):
        start_time = time.time()

        img = img_queue.get()
        #with img_queue.mutex:
        #    img_queue.queue.clear()

        class_ids = []
        confidences = []
        boxes = []
    
        if(img is not None):
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
                        center_x = int(detection[0] * w_h[0])
                        center_y = int(detection[1] * w_h[1])
                        w = int(detection[2] * w_h[0])
                        h = int(detection[3] * w_h[1])
                        x = center_x - w / 2
                        y = center_y - h / 2
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])
    
            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
            fps = frame_counter * 1.0 / (time.time() - start_time)
    
            for i in indices:
                i = i[0]
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                draw_prediction(img, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
            save_frame_buffer.append(img)
    
            print(fps, " fps")

    for i in range(len(save_frame_buffer)):
        filename = "out/" + str(i) + ".jpg"
        cv2.imwrite(filename, save_frame_buffer[i])
        print(".", end="", flush=True)

    print("save frame complete")
    import os
    os._exit(1)


if __name__ == "__main__":
    img_que = queue.LifoQueue()

    cap = cv2.VideoCapture("data/test.mp4")
    #cap = cv2.VideoCapture()

    ret,img=cap.read()
    width = img.shape[1] 
    height = img.shape[0] 

    Thread(target=capture, args=(img_que,cap)).start()
    Thread(target=inference, args=(img_que, (width, height))).start()
