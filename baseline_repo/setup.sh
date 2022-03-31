# # """### **Dataset Download** """

BASE_PATH=/home/ubuntu/team-samosa-tvqa/baseline_repo

wget https://tvqa.cs.unc.edu/files/tvqa_qa_release.tar.gz

wget https://tvqa.cs.unc.edu/files/tvqa_subtitles.tar.gz

tar xzf tvqa_qa_release.tar.gz

tar xzf tvqa_subtitles.tar.gz

wget https://tvqa.cs.unc.edu/files/tvqa_plus_annotations.tar.gz  -P ./tvqa_plus/
tar -xf ${BASE_PATH}/tvqa_plus/tvqa_plus_annotations.tar.gz -C ${BASE_PATH}/tvqa_plus/

wget https://tvqa.cs.unc.edu/files/tvqa_plus_annotations_preproc_with_test.tar.gz  -P ./tvqa_plus/
tar -xf ${BASE_PATH}/tvqa_plus/tvqa_plus_annotations_preproc_with_test.tar.gz -C ${BASE_PATH}/tvqa_plus/

wget https://tvqa.cs.unc.edu/files/tvqa_plus_subtitles.tar.gz  -P ./tvqa_plus/
tar -xf ${BASE_PATH}/tvqa_plus/tvqa_plus_subtitles.tar.gz -C ${BASE_PATH}/tvqa_plus/

wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
unzip -qqq ${BASE_PATH}/glove.6B

pip install pysrt
pip install transformers datasets
