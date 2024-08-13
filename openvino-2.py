import cv2 as cv
import os
from ultralytics import YOLO
import cpuinfo
from concurrent.futures import ThreadPoolExecutor

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


rtsp_url = 'rtsp://admin:Pac1nk0!@158.140.176.70:9292/Streaming/Channels/101'


model = YOLO('models/yolov8n_openvino_model')
conf_threshold = 0.3 

# Target class ID
target_class_ids = {0, 2, 3, 5, 7}  # Person, Car, Motorcycle, Bus, Truck


box_width, box_height = 1080, 720
line_y = 360  


def get_bbox_info(box):
    x1, y1, x2, y2, score, class_id = box
    return int(x1), int(y1), int(x2), int(y2), int(class_id), score

def bbox_centroid(x1, x2, y1, y2, frame):
    x_point = int(x1 + ((x2 - x1) / 2))
    y_point = int(y1 + ((y2 - y1) / 2))
    cv.circle(frame, (x_point, y_point), 2, (255, 255, 255), 2, cv.LINE_8)
    return x_point, y_point

def bbox_rectangle(x1, y1, x2, y2, frame):
    cv.rectangle(frame, (x1, y1), (x2, y2), (191, 64, 191), 3, cv.LINE_8)

def bbox_class_id_label(x1, y1, x2, y2, frame, class_id):
    cv.rectangle(frame, (x1, y1), (x1 + 30, y1 - 15), (255, 255, 255), -1, cv.LINE_8)
    cv.putText(frame, 'ID:{}'.format(class_id), (x1 + 5, y1 - 3), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv.LINE_AA)

def display_fps(frame, fps, cpu_type=''):
    font_scale = 0.6
    font = cv.FONT_HERSHEY_SIMPLEX
    fps_text = 'FPS: {}'.format(round(fps, 2))
    (text_width, text_height) = cv.getTextSize(fps_text, font, fontScale=font_scale, thickness=1)[0]
    height, width = frame.shape[:2]
    x_position, y_position = int(0.1 * width), int(0.1 * height)
    padding = 10

    rect_coord = ((x_position, y_position), (x_position + text_width + padding, y_position - text_height - padding))
    cv.rectangle(frame, rect_coord[0], rect_coord[1], (0, 0, 0), -1, cv.LINE_8)
    text_coord = (x_position + int(padding / 2), y_position - int(padding / 2))
    cv.putText(frame, fps_text, text_coord, font, fontScale=font_scale, color=(255, 255, 255), thickness=2)

    if cpu_type:
        cpu_text = 'CPU: {}'.format(cpu_type)
        (text_width, text_height) = cv.getTextSize(cpu_text, font, fontScale=font_scale, thickness=1)[0]
        y_position += text_height + padding
        rect_coord = ((x_position, y_position), (x_position + text_width + padding, y_position - text_height - padding))
        cv.rectangle(frame, rect_coord[0], rect_coord[1], (0, 0, 0), -1, cv.LINE_8)
        text_coord = (x_position + int(padding / 2), y_position - int(padding / 2))
        cv.putText(frame, cpu_text, text_coord, font, fontScale=font_scale, color=(255, 255, 255), thickness=2)

def cpu_info():
    return cpuinfo.get_cpu_info()['brand_raw']

def process_frame(frame, results):
    height, width = frame.shape[:2]
    x_center = width // 2
    y_center = height // 2
    x1 = x_center - box_width // 2
    y1 = y_center - box_height // 2
    x2 = x_center + box_width // 2
    y2 = y_center + box_height // 2

    
    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


    cv.line(frame, (x1, y1 + line_y), (x2, y1 + line_y), (0, 0, 255), 2)

    for result in results:
        for box in result.boxes.data.tolist():
            x1_box, y1_box, x2_box, y2_box, class_id, score = get_bbox_info(box)
            if class_id in target_class_ids and score >= conf_threshold:
                if x1 <= x1_box <= x2 and y1 <= y1_box <= y2:
                    bbox_rectangle(x1_box, y1_box, x2_box, y2_box, frame)
                    bbox_class_id_label(x1_box, y1_box, x2_box, y2_box, frame, class_id)
                    x_centroid, y_centroid = bbox_centroid(x1_box, x2_box, y1_box, y2_box, frame)
                    if y_centroid >= y1 + line_y:
                        print("Kendaraan melewati garis Y")  # Replace with your counting logic
    return frame

def main():
    cap = cv.VideoCapture(rtsp_url)
    cpu_type = cpu_info()

    with ThreadPoolExecutor(max_workers=2) as executor:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv.resize(frame, (1920, 1080))

            future = executor.submit(model, frame, stream=True)
            results = future.result()
            frame = process_frame(frame, results)

            cv.imshow('img', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
