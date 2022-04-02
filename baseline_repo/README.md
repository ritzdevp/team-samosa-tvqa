1) Set appropriate paths in CONSTANTS.py and setup.sh

2) --AWS RUN--
   Do 
   1) conda activate pytorch_p38
   2) run main.py


Download RESNET Data:

!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1m8bC4lefQsP2tRhMLAaiy0AVuBXZtegc' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=1m8bC4lefQsP2tRhMLAaiy0AVuBXZtegc" -O tvqa_imagenet_resnet101_pool5_hq.tar.gz && rm -rf /tmp/cookies.txt
