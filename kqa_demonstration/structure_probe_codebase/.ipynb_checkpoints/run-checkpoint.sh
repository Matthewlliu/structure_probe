LF_NAME='kopl'
MODEL='text-davinci-003'

/Users/matthewliu/anaconda3/envs/torch/bin/python main.py \
    --start_id 40000 \
    --augment_size 94376 \
    --save_step 5000 \
    --model_name ${MODEL} \
    --model_dir "data/api_keys_16-2.txt" \
    --spare_keys "data/api_keys_16-3.txt" \
    --logic_forms ${LF_NAME} \
    --data_dir "cache/${LF_NAME}/dataset/train.json" \
    --cache_dir "cache/${LF_NAME}/split_by_function/functions_ind.jsonl" \
    --output_dir "result/${LF_NAME}/{}" \
    --topk 50 \
    --topp 0.9 \
    --beam_size 10 \
    --batch_size 4 \
    --temperature 1 \
    --strategy beam_sample \
    --if_lf2nl 