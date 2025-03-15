#!/bin/bash
source activate LLMSeg_cp310
export PYTHONUNBUFFERED=1
python tool/merge_signle_video.py
