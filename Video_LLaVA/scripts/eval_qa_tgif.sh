

GPT_Zero_Shot_QA="/data/media/disk/jpf/GPT_Zero_Shot_QA"
output_name="Video-LLaVA-7B"
pred_path="${GPT_Zero_Shot_QA}/TGIF_Zero_Shot_QA/${output_name}/merge.jsonl"
output_dir="${GPT_Zero_Shot_QA}/TGIF_Zero_Shot_QA/${output_name}/gpt3.5-0.0"
output_json="${GPT_Zero_Shot_QA}/TGIF_Zero_Shot_QA/${output_name}/results.json"
api_key="sk-V6SeXEQWgCqg8WyzB3F102F771B742Bf8e96602a1aBe3c5b"
api_base="https://vip.apiyi.com/v1"
num_tasks=8



python3 videollava/eval/video/eval_video_qa.py \
    --pred_path ${pred_path} \
    --output_dir ${output_dir} \
    --output_json ${output_json} \
    --api_key ${api_key} \
    --api_base ${api_base} \
    --num_tasks ${num_tasks}
