. ./config.sh
# scp -P $AI_PORT -r "$KITTI_DATA" ubuntu@140.116.8.201:$AI_KITTI_DATA
scp -P $AI_PORT -r "$FLY_DATA" ubuntu@140.116.8.201:$AI_FLY_DATA
