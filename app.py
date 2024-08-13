import cv2
from flask import Flask, Response, redirect, url_for, request, jsonify
import torch
from ultralytics import YOLO
import threading
import time
import numpy as np
import uuid
import mysql.connector
from collections import defaultdict

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = YOLO('yolov8n.pt').to(device)
print(f"Model loaded on device: {device}")

vehicle_id = 0
detected_vehicles = {}
last_positions = {}
temporary_person_counts = 0
last_person_positions = defaultdict(lambda: (0, 0))

vehicle_counts_storage = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
total_count_storage = 0

line_position = {'y': 25}  

monitoring_data = {
    'vehicle_counts': vehicle_counts_storage,
    'total_count': total_count_storage,
    'person_count': temporary_person_counts
}

monitoring_codes_data = {}

db_config = {
    'user': 'root',
    'password': '',
    'host': 'localhost',
    'database': 'monitoring_db'
}

def create_monitoring_table(monitoring_code):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    
    table_name = f"monitoring_data_{monitoring_code}"
    
    query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id INT PRIMARY KEY,
        car_count INT DEFAULT 0,
        motorcycle_count INT DEFAULT 0,
        bus_count INT DEFAULT 0,
        truck_count INT DEFAULT 0,
        total_vehicle_count INT DEFAULT 0,
        person_count INT DEFAULT 0,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    )
    """
    
    cursor.execute(query)
    cursor.execute(f"INSERT IGNORE INTO {table_name} (id) VALUES (1)")
    
    conn.commit()
    cursor.close()
    conn.close()

def save_monitoring_data_to_db(monitoring_code, vehicle_counts, total_count, person_count):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    
    table_name = f"monitoring_data_{monitoring_code}"
    
    query = f"""
    UPDATE {table_name} SET
        car_count = %s,
        motorcycle_count = %s,
        bus_count = %s,
        truck_count = %s,
        total_vehicle_count = %s,
        person_count = %s
    WHERE id = 1
    """
    
    cursor.execute(query, (
        vehicle_counts['car'], 
        vehicle_counts['motorcycle'], 
        vehicle_counts['bus'], 
        vehicle_counts['truck'], 
        total_count, 
        person_count
    ))
    
    conn.commit()
    cursor.close()
    conn.close()

def count_objects(frame, model, monitoring_code):
    global vehicle_id, total_count_storage, temporary_person_counts, monitoring_data
    results = model(frame)
    current_vehicles = {}
    current_persons = {}
    detection_threshold = 50

    frame_height, frame_width = frame.shape[:2]
    box_x1, box_y1 = int(frame_width * 0.1), int(frame_height * 0.1)
    box_x2, box_y2 = int(frame_width * 0.9), int(frame_height * 0.9)
    line_position_y = box_y1 + int((box_y2 - box_y1) / 2)  

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            label = None

            if class_id in [2, 3, 5, 7]:
                if class_id == 2:
                    label = 'Car'
                elif class_id == 3:
                    label = 'Motorcycle'
                elif class_id == 5:
                    label = 'Bus'
                elif class_id == 7:
                    label = 'Truck'

                x1, y1, x2, y2 = box.xyxy[0]
                vehicle_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                if box_x1 < vehicle_center[0] < box_x2 and box_y1 < vehicle_center[1] < box_y2:
                    found = False
                    vid = None
                    for vid, vcenter in detected_vehicles.items():
                        distance = np.linalg.norm(np.array(vehicle_center) - np.array(vcenter))
                        if distance < detection_threshold:
                            current_vehicles[vid] = vehicle_center
                            found = True
                            break

                    if not found:
                        vehicle_id += 1
                        vid = vehicle_id
                        current_vehicles[vid] = vehicle_center
                        last_positions[vid] = vehicle_center

                    if vid is not None and last_positions[vid][1] < line_position_y and vehicle_center[1] >= line_position_y:
                        vehicle_counts_storage[label.lower()] += 1
                        total_count_storage += 1

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(frame, f'{label}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

            elif class_id == 0:
                x1, y1, x2, y2 = box.xyxy[0]
                person_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                if box_x1 < person_center[0] < box_x2 and box_y1 < person_center[1] < box_y2:
                    found = False
                    pid = None
                    for pid, pcenter in last_person_positions.items():
                        distance = np.linalg.norm(np.array(person_center) - np.array(pcenter))
                        if distance < detection_threshold:
                            current_persons[pid] = person_center
                            found = True
                            break

                    if not found:
                        pid = len(last_person_positions) + 1
                        current_persons[pid] = person_center
                        last_person_positions[pid] = person_center

                    if pid is not None and last_positions[pid][1] < line_position_y and person_center[1] >= line_position_y:
                        temporary_person_counts += 1

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, 'Person', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    detected_vehicles.update(current_vehicles)
    for vid in current_vehicles.keys():
        last_positions[vid] = current_vehicles[vid]

    last_person_positions.update(current_persons)

    monitoring_data = {
        'vehicle_counts': vehicle_counts_storage,
        'total_count': total_count_storage,
        'person_count': temporary_person_counts
    }

    save_monitoring_data_to_db(monitoring_code, vehicle_counts_storage, total_count_storage, temporary_person_counts)

    return vehicle_counts_storage, total_count_storage, temporary_person_counts

class VideoCamera:
    def __init__(self, rtsp_url=''):
        self.rtsp_url = rtsp_url
        self.capture = None
        self.frame = None
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        self.set_rtsp_url(rtsp_url)

    def __del__(self):
        if self.capture:
            self.capture.release()

    def get_frame(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            else:
                return None

    def set_rtsp_url(self, rtsp_url):
        with self.lock:
            self.rtsp_url = rtsp_url
            if self.capture:
                self.capture.release()
            self.capture = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            if not self.capture.isOpened():
                print(f"Error: Unable to open video stream at {rtsp_url}.")
                return False
            print(f"Video stream opened successfully at {rtsp_url}.")
            return True

    def update(self):
        while True:
            if self.capture:
                ret, frame = self.capture.read()
                if ret:
                    with self.lock:
                        self.frame = frame
                else:
                    print("Error: Unable to read frame. Retrying...")
                    time.sleep(1)
                    self.capture.release()
                    self.capture = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            else:
                time.sleep(0.01)

camera = VideoCamera()

def generate_monitoring_code():
    monitoring_code = str(uuid.uuid4())[:8]
    monitoring_codes_data[monitoring_code] = {
        'vehicle_counts': {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0},
        'total_count': 0,
        'person_count': 0,
        'line_position': line_position['y']
    }
    create_monitoring_table(monitoring_code)
    return monitoring_code

def generate_frames(monitoring_code):
    fps = 25  
    sleep_interval = 1 / fps

    while True:
        start_time = time.time()
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        vehicle_counts, total_count, persons = count_objects(frame, model, monitoring_code)

        frame_height, frame_width = frame.shape[:2]
        box_x1, box_y1 = int(frame_width * 0.1), int(frame_height * 0.1)
        box_x2, box_y2 = int(frame_width * 0.9), int(frame_height * 0.9)
        line_position_y = box_y1 + int((box_y2 - box_y1) / 2)

        # Draw smaller box
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 255), 2)
        cv2.line(frame, (box_x1, line_position_y), (box_x2, line_position_y), (0, 255, 255), 2)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 120), (460, 385), (0, 0, 0), -1)
        alpha = 0.5
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        cv2.putText(frame, 'Car: {}'.format(vehicle_counts['car']), (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, 'Motorcycle: {}'.format(vehicle_counts['motorcycle']), (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, 'Bus: {}'.format(vehicle_counts['bus']), (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, 'Truck: {}'.format(vehicle_counts['truck']), (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, 'Total Vehicles: {}'.format(total_count), (10, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, 'Persons: {}'.format(persons), (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        end_time = time.time()
        elapsed_time = end_time - start_time
        sleep_time = max(sleep_interval - elapsed_time, 0)
        time.sleep(sleep_time)

@app.route('/set_line_position/<monitoring_code>', methods=['POST'])
def set_line_position(monitoring_code):
    global line_position
    y = request.json.get('y')
    if y is not None and isinstance(y, int):
    
        line_position['y'] = y
        
 
        if monitoring_code in monitoring_codes_data:
            monitoring_codes_data[monitoring_code]['line_position'] = y

        return jsonify({'status': 'success', 'message': f'Line position set to {y} for monitoring code {monitoring_code}'}), 200
    else:
        return jsonify({'status': 'error', 'message': 'Invalid y position'}), 400

@app.route('/set_rtsp_url', methods=['POST'])
def set_rtsp_url():
    rtsp_url = request.json.get('rtsp_url')

    if rtsp_url:
        monitoring_code = generate_monitoring_code() 

        if camera.set_rtsp_url(rtsp_url):
            monitoring_codes_data[monitoring_code] = monitoring_data
            return jsonify({'status': 'success', 'message': f'RTSP URL set to {rtsp_url}', 'monitoring_code': monitoring_code}), 200
        else:
            return jsonify({'status': 'error', 'message': 'Failed to set RTSP URL'}), 500
    else:
        return jsonify({'status': 'error', 'message': 'Invalid RTSP URL'}), 400

@app.route('/get_monitoring_data/<monitoring_code>', methods=['GET'])
def get_monitoring_data(monitoring_code):
    data = monitoring_codes_data.get(monitoring_code)
    if data is not None:
    
        data['vehicle_counts'] = monitoring_data['vehicle_counts']
        data['total_count'] = monitoring_data['total_count']
        data['person_count'] = monitoring_data['person_count']
        return jsonify(data), 200
    else:
        return jsonify({'status': 'error', 'message': 'Monitoring code not found'}), 404

@app.route('/active_monitoring_codes', methods=['GET'])
def active_monitoring_codes():
    return jsonify(list(monitoring_codes_data.keys())), 200

@app.route('/monitoring/<monitoring_code>')
def monitoring(monitoring_code):
    if monitoring_code not in monitoring_codes_data:
        return jsonify({'status': 'error', 'message': 'Invalid monitoring code'}), 403

    return Response(generate_frames(monitoring_code), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return redirect(url_for('active_monitoring_codes'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
