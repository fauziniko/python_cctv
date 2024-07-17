import cv2
from flask import Flask, Response, redirect, url_for, request, jsonify
import torch
from ultralytics import YOLO
import threading
import time
import numpy as np
import uuid  # Untuk menghasilkan UUID

app = Flask(__name__)

# Load YOLO model
model = YOLO('yolov8s.pt')

# Initialize vehicle ID counter, detected vehicles, and person counts
vehicle_id = 0
detected_vehicles = {}
last_positions = {}
temporary_person_counts = 0
last_person_positions = {}

# Temporary storage for vehicle counts and total count
vehicle_counts_storage = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
total_count_storage = 0

# Counting line position (default)
line_position = {'y': 500}

# Global variable to store the latest monitoring data
monitoring_data = {
    'vehicle_counts': vehicle_counts_storage,
    'total_count': total_count_storage,
    'person_count': temporary_person_counts
}

# Function to count vehicles and persons, and draw bounding boxes with IDs
def count_objects(frame, model):
    global vehicle_id, total_count_storage, temporary_person_counts, monitoring_data
    results = model(frame)
    current_vehicles = {}
    current_persons = {}
    detection_threshold = 50  # Distance threshold for matching vehicles

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            label = None

            # Detect vehicles
            if class_id in [2, 3, 5, 7]:  # Car, Motorcycle, Bus, Truck
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

                found = False
                vid = None
                for vid, vcenter in detected_vehicles.items():
                    distance = np.linalg.norm(np.array(vehicle_center) - np.array(vcenter))
                    if distance < detection_threshold:  # Same vehicle
                        current_vehicles[vid] = vehicle_center
                        found = True
                        break

                if not found:
                    vehicle_id += 1
                    vid = vehicle_id
                    current_vehicles[vid] = vehicle_center
                    last_positions[vid] = vehicle_center

                if vid is not None and last_positions[vid][1] < line_position['y'] and vehicle_center[1] >= line_position['y']:
                    vehicle_counts_storage[label.lower()] += 1
                    total_count_storage += 1

                # Draw bounding box with vehicle label (without ID)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f'{label}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

            # Detect persons
            elif class_id == 0:  # Assuming class ID 0 corresponds to persons
                x1, y1, x2, y2 = box.xyxy[0]
                person_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                # Check if person crosses line from above
                if person_center[1] >= line_position['y'] and last_person_positions.get(person_center, None) is not None:
                    if last_person_positions[person_center][1] < line_position['y']:
                        temporary_person_counts += 1

                # Draw bounding box for persons
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, 'Person', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                current_persons[person_center] = person_center

    # Update detected vehicles and their last positions
    detected_vehicles.update(current_vehicles)
    for vid in current_vehicles.keys():
        last_positions[vid] = current_vehicles[vid]

    # Update last positions for persons
    last_person_positions.update(current_persons)

    # Update monitoring data
    monitoring_data = {
        'vehicle_counts': vehicle_counts_storage,
        'total_count': total_count_storage,
        'person_count': temporary_person_counts
    }

    return vehicle_counts_storage, total_count_storage, temporary_person_counts

class VideoCamera:
    def __init__(self, rtsp_url=''):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def __del__(self):
        if self.cap:
            self.cap.release()

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def set_rtsp_url(self, rtsp_url):
        with self.lock:
            self.rtsp_url = rtsp_url
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(rtsp_url)
            if not self.cap.isOpened():
                print("Error: Unable to open video stream.")
                self.cap = None
                return None
            else:
                return generate_monitoring_code()  # Mengembalikan monitoring_code setelah berhasil mengatur RTSP URL

    def update(self):
        while True:
            if self.cap:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Unable to read frame.")
                    time.sleep(1)
                    continue
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

camera = VideoCamera()

def generate_monitoring_code():
    # Menghasilkan kode monitoring menggunakan 8 karakter pertama dari UUID
    return str(uuid.uuid4())[:8]

def generate_frames():
    frame_count = 0
    process_interval = 1  # frame rate
    fps = 30
    sleep_interval = 1 / fps

    last_processed_frame = None

    while True:
        start_time = time.time()
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        frame_count += 1

        if frame_count % process_interval == 0:
            vehicle_counts, total_count, persons = count_objects(frame, model)

            # Draw the counting line
            cv2.line(frame, (0, line_position['y']), (frame.shape[1], line_position['y']), (0, 255, 255), 2)

            # Display the result on the frame
            y_offset = 120
            cv2.putText(frame, f"Cars: {vehicle_counts['car']}", (10, 30 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(frame, f"Motorcycles: {vehicle_counts['motorcycle']}", (10, 70 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(frame, f"Buses: {vehicle_counts['bus']}", (10, 110 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(frame, f"Trucks: {vehicle_counts['truck']}", (10, 150 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(frame, f'Total Vehicles: {total_count}', (10, 190 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(frame, f'Total Persons: {persons}', (10, 230 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            last_processed_frame = frame.copy()

        if last_processed_frame is not None:
            frame_to_show = last_processed_frame
        else:
            frame_to_show = frame

        ret, buffer = cv2.imencode('.jpg', frame_to_show)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_line_position', methods=['POST'])
def set_line_position():
    global line_position
    y = request.json.get('y')
    if y is not None and isinstance(y, int):
        line_position['y'] = y
        return jsonify({'status': 'success', 'message': f'Line position set to {y}'}), 200
    else:
        return jsonify({'status': 'error', 'message': 'Invalid y position'}), 400

@app.route('/set_rtsp_url', methods=['POST'])
def set_rtsp_url():
    rtsp_url = request.json.get('rtsp_url')
    if rtsp_url:
        monitoring_code = camera.set_rtsp_url(rtsp_url)
        if monitoring_code:
            return jsonify({'status': 'success', 'message': f'RTSP URL set to {rtsp_url}', 'monitoring_code': monitoring_code}), 200
        else:
            return jsonify({'status': 'error', 'message': 'Failed to set RTSP URL'}), 500
    else:
        return jsonify({'status': 'error', 'message': 'Invalid RTSP URL'}), 400

@app.route('/get_monitoring_data', methods=['GET'])
def get_monitoring_data():
    return jsonify(monitoring_data), 200

@app.route('/monitoring/<monitoring_code>')
def monitoring(monitoring_code):
    # Implementasi untuk menampilkan tampilan kamera berdasarkan monitoring_code
    # Di sini Anda bisa mengembalikan tampilan HTML atau mengirimkan gambar streaming
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return redirect(url_for('video_feed'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
