#!/bin/bash

python3 ahus_predict.py --load_path "/workspace/inputs" --save_path "/workspace/outputs" --model_checkpoint "/workspace/weights.pth" --model_type "ahus_model_rope_mixed" --segmenter_type "original"
