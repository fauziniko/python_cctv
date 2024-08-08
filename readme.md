## Install Yang diperlukan ##

pip3 install ultralytics opencv-python-headless flask ffmpeg-python

## Menjalankan code ##

python app.py

## Membuat docker ##

docker build -t vehicle_detection_app .


## Menjalankan Container ##
docker run -p 5000:5000 vehicle_detection_app

## Menjalankan di docker 

# Log in to Docker Hub
docker login

# Tag the image
docker tag vehicle_detection_app your_username/vehicle_detection_app:latest

# Push the image to Docker Hub
docker push your_username/vehicle_detection_app:latest

#Convert model dari .onnx ke dalam .xml/.bin#
mo --input_model model.onnx

# Pull the image from Docker Hub
docker pull your_username/vehicle_detection_app:latest

# Run the container
docker run -p 5000:5000 your_username/vehicle_detection_app:latest
