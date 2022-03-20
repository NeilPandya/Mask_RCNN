FROM tensorflow/tensorflow:1.15.0-gpu-py3
RUN apt-get update
RUN apt-get install apt-utils -y
RUN apt-get install nano git ffmpeg libsm6 libxext6 -y
RUN apt-get update
RUN git clone https://github.com/NeilPandya/Mask_RCNN.git

RUN cd /Mask_RCNN
COPY mask_rcnn_coco.h5 /Mask_RCNN

RUN pip install --upgrade pip
RUN pip uninstall networkx decorator idna -y
RUN pip install -r /Mask_RCNN/requirements.txt
RUN ./getDatasets.sh
RUN ./getCOCOweights.sh

RUN apt update
RUN chmod 777 -R /Mask_RCNN

EXPOSE 8888/tcp
EXPOSE 6006/tcp
EXPOSE 6007/tcp
EXPOSE 6008/TCP
EXPOSE 6009/TCP
EXPOSE 6010/TCP