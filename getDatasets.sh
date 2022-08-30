#!/bin/bash

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1xlzzjHlZViof6s8W_EHHLHZkiLN4rAST' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1xlzzjHlZViof6s8W_EHHLHZkiLN4rAST" -O datasets.zip && rm -rf /tmp/cookies.txt
unzip -q -y datasets.zip
rm datasets.zip
chmod 777 -R datasets
