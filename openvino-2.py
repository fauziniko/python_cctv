from ultralytics import YOLO
import cv2 as cv
import os

import helper

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


rtsp_url = 'rtsp://admin:Pac1nk0!@158.140.176.70:9292/Streaming/Channels/101'
cap = cv.VideoCapture(rtsp_url)

model = YOLO('yolov8n_openvino_model/')

conf_threshold = 0.3 

frame_counter = 0
frames = []


while cap.isOpened():
    
    success, frame = cap.read()
    
    if success:

        results = model(frame, stream=True)
        
        for result in results:

            for box in result.boxes.data.tolist():
                

                x1, y1, x2, y2, class_id, score = helper.get_bbox_info(box)


                if class_id == 2 and score >= conf_threshold:
                    
 
                    helper.bbox_rectangle(x1, y1, x2, y2, frame)
                    helper.bbox_class_id_label(x1, y1, x2, y2, frame, class_id)
                    

                    x_centroid, y_centroid = helper.bbox_centroid(x1, x2, y1, y2, frame)
                
    cv.imshow('img', frame)
    print('Showing -> Frame: {}'.format(frame_counter))
    frame_counter += 1
    frames.append(frame_counter)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        cv.destroyAllWindows()
        break
        
cap.release()
cv.destroyAllWindows()