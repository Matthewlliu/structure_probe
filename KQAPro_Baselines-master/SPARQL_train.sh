INPUT='/home/ljx/new_cache_server32_0411/KQAPro/dataset'
PROCESSED='/home/ljx/new_cache_server32_0411/KQAPro/sparql_processed_test'
CHECKPOINT='/data/MODELS/bart-base'
OUTPUT='/data/ljx/result/probeLLM/sparql/bart-base_test_230601'

CUDA_VISIBLE_DEVICES=3 python -m Bart_SPARQL.train \
    --input_dir ${PROCESSED} \
    --output_dir "${OUTPUT}/checkpoint" \
    --save_dir "${OUTPUT}/logs" \
    --model_name_or_path ${CHECKPOINT} \
    --num_train_epochs 3 \
    --batch_size 16 \
    --learning_rate 1e-4