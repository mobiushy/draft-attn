#!/bin/bash

python -m llava.eval.model_vqa \
    --model-path /data/media/disk/jpf/llava-v1.5-7b \
    --question-file ./playground/data/eval/nocaps/nocaps_val_4500_captions.jsonl \
    --image-folder ./playground/data/eval/nocaps/images \
    --answers-file ./playground/data/eval/nocaps/answers/llava-v1.5-7b_topr_test.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/nocaps/nocaps_val_4500_captions.jsonl \
    --result-file ./playground/data/eval/nocaps/answers/llava-v1.5-7b_topr_test.jsonl \
    --result-upload-file ./playground/data/eval/nocaps/answers_upload/llava-v1.5-7b_topr_test.json
