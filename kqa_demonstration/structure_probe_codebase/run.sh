LF_NAME='sparql'
MODEL='flan-t5-xxl'

CUDA_VISIBLE_DEVICES=1 python main.py \
    --start_id 0 \
    --augment_size 44337 \
    --save_step 5000 \
    --model_name ${MODEL} \
    --model_dir "/data3/MODELS/${MODEL}" \
    --spare_keys "data/api_keys_10-2.txt" \
    --logic_forms ${LF_NAME} \
    --data_dir "/home/ljx/new_cache_server32_0411/kqa_demonstration/structure_probe_codebase/cache/${LF_NAME}/dataset/grailqa_v1.0_train.json" \
    --cache_dir "cache/${LF_NAME}/split_by_function/functions_ind.jsonl" \
    --output_dir "/data1/ljx/result/probeLLM/${LF_NAME}/{}" \
    --topk 50 \
    --topp 0.9 \
    --beam_size 5 \
    --batch_size 4 \
    --temperature 1 \
    --strategy beam_sample \
    --if_lf2nl 
    #--spare_keys "data/api_keys_10-2.txt" \gpt-j-sharded-fp16 \ 94376 \ 44337 \ 20874