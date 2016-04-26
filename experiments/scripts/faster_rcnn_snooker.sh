#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_alt_opt.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is only pascal_voc for now
#
# Example:
# ./experiments/scripts/faster_rcnn_alt_opt.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
DATASET=$3

array=( $@ )
len=${#array[@]}

case $DATASET in
  snooker)
    TRAIN_IMDB="snooker_train"
    TEST_IMDB="snooker_test"
    PT_DIR="snooker_net"
    ;;
  snooker_debug)
    TRAIN_IMDB="snooker_debug_train"
    TEST_IMDB="snooker_debug_test"
    PT_DIR="snooker_net"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/faster_rcnn_alt_opt_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_faster_rcnn_alt_opt.py --gpu ${GPU_ID} \
  --net_name ${NET} \
  --imdb ${TRAIN_IMDB} \
  --weights data/imagenet_models/${NET}.v2.caffemodel \
  --cfg experiments/cfgs/faster_rcnn_snooker.yml 

set +x
NET_FINAL=`grep "Final model:" ${LOG} | awk '{print $3}'`
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/faster_rcnn_alt_opt/faster_rcnn_test.pt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_snooker.yml
