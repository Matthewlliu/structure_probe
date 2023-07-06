INPUT='/home/ljx/new_cache_server32_0411/GrailQA_data'
PROCESSED="${INPUT}/bart-base_processed"
#CHECKPOINT='/data/ljx/cpt/bart-base'
OUTPUT='/data/ljx/result/probeLLM/sparql/grailqa_bart-base_human_230706'

CUDA_VISIBLE_DEVICES=3 python -m Bart_SPARQL.train \
    --input_dir ${PROCESSED} \
    --output_dir "${OUTPUT}/checkpoint" \
    --save_dir "${OUTPUT}/logs" \
    --model_name_or_path "${PROCESSED}" \
    --num_train_epochs 10 \
    --batch_size 8 \
    --learning_rate 1e-5  