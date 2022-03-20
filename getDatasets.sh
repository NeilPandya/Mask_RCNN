#!/bin/bash

cd /darknet
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1xlzzjHlZViof6s8W_EHHLHZkiLN4rAST' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nzbagwoCKMHCY_Jvcc2pfQVHnjFx66FY" -O datasets.zip && rm -rf /tmp/cookies.txt
unzip -q datasets.zip
rm /Mask_RCNN/datasets.zip
chmod 777 -R /Mask_RCNN/datasets
