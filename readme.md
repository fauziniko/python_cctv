pip3 install ultralytics opencv-python-headless flask ffmpeg-python


docker build -t vehicle_detection_app .


docker run -p 5000:5000 vehicle_detection_app

