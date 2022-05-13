import cv2
import numpy as np
import time


def capture_to_buffer():
    cap = cv2.VideoCapture("data/test.mp4")
    frame_buffer = []
    width = 0
    height = 0

    try:
        for i in range(5):
            print(".", end="", flush=True)
            ret, img = cap.read()
            if(ret):
                frame_buffer.append(img)
                width = img.shape[1]
                height = img.shape[0]
            else:
                break
    except Exception as e:
        print(e)

    return frame_buffer, width, height


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    with open("data/coco.names", 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    label = str(classes[class_id])
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), (0, 128, 128), 5)
    cv2.putText(img, label, (x-20, y-20), cv2.FONT_HERSHEY_PLAIN, 5, (0, 128, 128), 2)


def main():

    frame_buffer, Width, Height = capture_to_buffer()
    print("frame buffer complete", Width, Height)

    #net = cv2.dnn.readNet("data/yolov3.weights", "data/yolov3.cfg")
    net = cv2.dnn.readNet("data/yolov3-tiny.weights", "data/yolov3-tiny.cfg")

    output_layers = net.getUnconnectedOutLayersNames()

    scale = 0.00392
    conf_threshold = 0.1
    nms_threshold = 0.4
    frame_counter = 0
    start_time = time.time()
    fps = 0.0

    save_frame_buffer = []

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

        print(".", end="", flush=True)

    print(fps, " fps")

    for i in range(len(save_frame_buffer)):
        filename = "out/" + str(i) + ".jpg"
        cv2.imwrite(filename, save_frame_buffer[i])
        print(".", end="", flush=True)

    print("save frame complete")


if __name__ == "__main__":
    main()
