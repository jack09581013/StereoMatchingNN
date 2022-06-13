. ./config.sh
mkdir $MODEL_DIR -p
sshpass -p $PASSWARD scp -P $AI_PORT -r ubuntu@140.116.8.201:$AI_MODEL_DIR $MODEL_DIR

