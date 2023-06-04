INPUT='/data/ljx/data/KQAPro/dataset'
PROCESSED='/data/ljx/data/KQAPro/processed'
CHECKPOINT='/data/ljx/cpt/bart-base'

OUTPUT='/data/ljx/result/kqa_pro/bart-base_program_6_7'

CUDA_VISIBLE_DEVICES=5 python -m Bart_Program.preprocess \
    --input_dir ${INPUT} \
    --output_dir ${PROCESSED} \
    --model_name_or_path ${CHECKPOINT}

cp "${INPUT}/kb.json" ${PROCESSED}

#CUDA_VISIBLE_DEVICES=5 python -m Bart_Program.train \
#    --input_dir ${PROCESSED} \
#    --output_dir "${OUTPUT}/checkpoint" \
#    --save_dir "${OUTPUT}/logs" \
#    --model_name_or_path ${CHECKPOINT}