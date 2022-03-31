# """### **Dataset Download** """

BASE_PATH=/content/drive/MyDrive/MMML

wget https://tvqa.cs.unc.edu/files/tvqa_qa_release.tar.gz -P ${BASE_PATH}

wget https://tvqa.cs.unc.edu/files/tvqa_subtitles.tar.gz -P ${BASE_PATH}

tar xzf ${BASE_PATH}/tvqa_qa_release.tar.gz -C ${BASE_PATH}

tar xzf ${BASE_PATH}/tvqa_subtitles.tar.gz  -C ${BASE_PATH}

wget https://tvqa.cs.unc.edu/files/tvqa_plus_annotations.tar.gz  -P ${BASE_PATH}/tvqa_plus/
tar -xf ${BASE_PATH}/tvqa_plus/tvqa_plus_annotations.tar.gz -C ${BASE_PATH}/tvqa_plus/

wget https://tvqa.cs.unc.edu/files/tvqa_plus_annotations_preproc_with_test.tar.gz  -P ${BASE_PATH}/tvqa_plus/
tar -xf ${BASE_PATH}/tvqa_plus/tvqa_plus_annotations_preproc_with_test.tar.gz -C ${BASE_PATH}/tvqa_plus/

wget https://tvqa.cs.unc.edu/files/tvqa_plus_subtitles.tar.gz  -P ${BASE_PATH}/tvqa_plus/
tar -xf ${BASE_PATH}/tvqa_plus/tvqa_plus_subtitles.tar.gz -C ${BASE_PATH}/tvqa_plus/

pip install pysrt
pip install transformers datasets
pip install wandb
