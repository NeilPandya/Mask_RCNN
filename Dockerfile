FROM tensorflow/tensorflow:1.15.0-gpu-py3
RUN apt-get update
RUN apt-get install nano git ffmpeg libsm6 libxext6 -y
RUN apt-get update
RUN git clone https://github.com/NeilPandya/Mask_RCNN.git

RUN cd /Mask_RCNN
COPY requirements.txt /Mask_RCNN
COPY video_demo.py /Mask_RCNN
COPY visualize_cv2.py /Mask_RCNN
COPY mask_rcnn_coco.h5 /Mask_RCNN
COPY samples /Mask_RCNN/samples

RUN chmod 777 -R /Mask_RCNN
RUN pip uninstall networkx decorator idna -y
RUN pip install -r /Mask_RCNN/requirements.txt
RUN apt update

EXPOSE 8888/tcp
EXPOSE 6006/tcp