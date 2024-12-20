#!/usr/bin/bash

export PYTHONPATH="${PYTHONPATH}:/home/mariopasc/Python/Projects/Coronary_Angiography_Detection"

python ./create_fold_info.py
python ./create_yolo_splits.py