#!/bin/bash
# Command used to train the best-performing Fresh_aug model on the TITAN RTX
yolo task=detect mode=train \
  data=/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/data/urbaning_12k/data.yaml \
  model=/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/models/finals/urbaning_fresh_12k_X/weights/best.pt \
  epochs=100 patience=25 batch=8 imgsz=640 cos_lr=True \
  mosaic=1.0 mixup=0.15 scale=0.4 translate=0.1 degrees=10.0 flipud=0.0 fliplr=0.5 \
  hsv_h=0.0 hsv_s=0.0 hsv_v=0.0 \
  close_mosaic=15 optimizer=AdamW lr0=0.0001 \
  name=urbaning_retrain_augmented_v2 \
  project=/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/models/finals
