## Install Yang diperlukan ##

pip3 install ultralytics opencv-python-headless flask ffmpeg-python

## Menjalankan code ##

python app.py

## Membuat docker ##

docker build -t vehicle_detection_app .


## Menjalankan Container ##
docker run -p 5000:5000 vehicle_detection_app

