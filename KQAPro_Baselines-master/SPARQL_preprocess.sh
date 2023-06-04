INPUT='/home/ljx/new_cache_server32_0411/KQAPro/dataset'
PROCESSED='/home/ljx/new_cache_server32_0411/KQAPro/sparql_processed_test'
CHECKPOINT='/data/MODELS/bart-base'

CUDA_VISIBLE_DEVICES=3 python -m Bart_SPARQL.preprocess \
    --input_dir ${INPUT} \
    --output_dir ${PROCESSED} \
    --model_name_or_path ${CHECKPOINT}

cp "${INPUT}/kb.json" ${PROCESSED}