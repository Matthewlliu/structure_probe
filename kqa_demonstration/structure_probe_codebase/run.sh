LF_NAME='kopl'
MODEL='code-davinci-002'

CUDA_VISIBLE_DEVICES=5 python main.py \
    --start_id 0 \
    --augment_size 94376 \
    --save_step 5000 \
    --model_name ${MODEL} \
    --model_dir "data/api_keys_10-2.txt" \
    --spare_keys "data/api_keys_10-2.txt" \
    --logic_forms ${LF_NAME} \
    --data_dir "/home/ljx/new_cache_server32_0411/kqa_demonstration/structure_probe_codebase/cache/${LF_NAME}/dataset/train.json" \
    --test_dir "/home/ljx/new_cache_server32_0411/kqa_demonstration/structure_probe_codebase/cache/${LF_NAME}/dataset/val.json" \
    --cache_dir "cache/${LF_NAME}/split_by_function/functions_ind.jsonl" \
    --output_dir "/data/ljx/result/probeLLM/${LF_NAME}/{}" \
    --topk 50 \
    --topp 0.9 \
    --beam_size 5 \
    --batch_size 1 \
    --temperature 1 \
    --strategy beam_sample \
    --demo_num 3 \
    --toy \
    --if_lf2nl #>out_kopl_davinci002_lf2nl.txt 2>&1 &
    
    #>out_kopl_davinci003_nl2lf_whole_300.txt 2>&1 &

    #--if_naive #>out_kopl_naive_60k2end.txt 2>&1 &
    #--if_naive # >out_kopl_naive_30k260k.txt 2>&1 &
    #--toy #>out_kopl_naive_30000.txt 2>&1 &
    #--spare_keys "data/api_keys_10-2.txt" \gpt-j-sharded-fp16 \ 94376 \ 44337 \ 20874
    #--if_naive \
    #--if_lf2nl \
    #--data_dir "/home/ljx/new_cache_server32_0411/kqa_demonstration/structure_probe_codebase/cache/${LF_NAME}/dataset/grailqa_v1.0_train.json" \
    #--data_dir "/home/ljx/semantic-parsing-dual-master/data/overnight/all_train.tsv" \