import cv2
from flask import Flask, Response, redirect, url_for, request, jsonify
import torch
from ultralytics import YOLO
import threading
import time
import numpy as np
import uuid

app = Flask(__name__)

# Load YOLO model with GPU support
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = YOLO('yolov8x.pt').to(device)
print(f"Model loaded on device: {device}")

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

# Store monitoring codes with their corresponding data
monitoring_codes_data = {}

# Function to count vehicles and persons, and draw bounding boxes with IDs
def count_objects(frame, model):
    global vehicle_id, total_count_storage, temporary_person_counts, monitoring_data
    results = model(frame)
    current_vehicles = {}
    current_persons = {}
    detection_threshold = 50  # Distance threshold for matching vehicles/persons

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

                found = False
                pid = None
                for pid, pcenter in last_person_positions.items():
                    distance = np.linalg.norm(np.array(person_center) - np.array(pcenter))
                    if distance < detection_threshold:  # Same person
                        current_persons[pid] = person_center
                        found = True
                        break

                if not found:
                    pid = len(last_person_positions) + 1  # Assign a new ID
                    current_persons[pid] = person_center
                    last_person_positions[pid] = person_center

                # Check if the person crosses the line
                if pid is not None and last_person_positions[pid][1] < line_position['y'] and person_center[1] >= line_position['y']:
                    temporary_person_counts += 1  # Count the person

                # Draw bounding box for persons
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, 'Person', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

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
                return generate_monitoring_code()  # Return monitoring_code after successfully setting RTSP URL

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
    # Generate monitoring code using first 8 characters of UUID
    monitoring_code = str(uuid.uuid4())[:8]
    monitoring_codes_data[monitoring_code] = monitoring_data  # Store initial data
    return monitoring_code

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

            # Draw a transparent background for text (rectangle with transparency)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 120), (460, 385), (0, 0, 0), -1)  # Black rectangle
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)  # Blend the overlay with the frame

            # Display the result on the frame
            y_offset = 140
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

@app.route('/set_line_position/<monitoring_code>', methods=['POST'])
def set_line_position(monitoring_code):
    global line_position
    y = request.json.get('y')
    if y is not None and isinstance(y, int):
        # Update the line position globally
        line_position['y'] = y
        
        # Update the line position in the monitoring codes data
        if monitoring_code in monitoring_codes_data:
            monitoring_codes_data[monitoring_code]['line_position'] = y

        return jsonify({'status': 'success', 'message': f'Line position set to {y} for monitoring code {monitoring_code}'}), 200
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

@app.route('/get_monitoring_data/<monitoring_code>', methods=['GET'])
def get_monitoring_data(monitoring_code):
    data = monitoring_codes_data.get(monitoring_code)
    if data is not None:
        # Return the latest vehicle counts and person count
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

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return redirect(url_for('active_monitoring_codes'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)