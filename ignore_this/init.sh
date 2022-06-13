. ./config.sh
ssh ubuntu@140.116.8.201 -p $AI_PORT << EOF
    mkdir /workspace/Dataset/pytorch -p    
    mkdir /workspace/Projects -p    
EOF
sshpass -p $PASSWARD scp -P $AI_PORT -r $LIB_DIR ubuntu@140.116.8.201:$AI_PROJECT_DIR
