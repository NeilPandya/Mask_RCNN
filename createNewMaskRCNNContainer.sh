#! /bin/bash

 # THIS SCRIPT WILL NOT WORK IF YOU DON'T HAVE A WEBCAM OR SIMILAR DEVICE
 # PLUGGED IN THAT YOUR SYSTEM HAS IDENTIFIES AS /dev/video0
 #
 # Change or delete this parameter to suit your needs; it was included for the
 # live video_demo.py.

xhost + && docker run \
	-i \
	-t \
	--name Mask_RCNN \
	-h Mask_RCNN \
	-u $(id -u):$(id -g) \
	-e DISPLAY \
	-e HOST_PERMS="$(id -u):$(id -g)" \
	--device /dev/video0/ \
	--gpus=all \
  -v /tmp/.X11-unix/:/tmp/.X11-unix \
	-v $PWD:/mnt \
	-p 8888:8888 \
	-p 6006:6006 \
	neilpandya/nsv:ISIC-Oral-Polyp-Mask_RCNN-tf1.15.0
