INPUT='/home/ljx/new_cache_server32_0411/GrailQA_data'
PROCESSED="${INPUT}/bart-base_processed"
CHECKPOINT='/data/ljx/cpt/bart-base'

CUDA_VISIBLE_DEVICES=3 python -m Bart_SPARQL.preprocess \
    --input_dir ${INPUT} \
    --output_dir ${PROCESSED} \
    --model_name_or_path ${CHECKPOINT} #>out_preprocess_0702.txt 2>&1 &