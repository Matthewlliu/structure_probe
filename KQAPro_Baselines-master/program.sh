#INPUT='/data/ljx/result/probeLLM/kopl/lf2nl_kopl_text-davinci-001_2023-03-27_94376'
INPUT='/home/ljx/new_cache_server32_0411/KQAPro/dataset'
PROCESSED='/data/ljx/data/KQAPro/davinci-001_lf2nl'
#CHECKPOINT='/data/ljx/cpt/new_download_bart_kqa'
CHECKPOINT='/data/MODELS/bart-base'
#CHECKPOINT='/data/ljx/result/para_model/bart-base-stage-5epochs_2022-06-07/checkpoints/checkpoint-18750'

#OUTPUT='/data/ljx/result/kqa_pro/bart-base_program_augonly_train_230307'

CUDA_VISIBLE_DEVICES=7 python -m Bart_Program.preprocess \
    --input_dir ${INPUT} \
    --output_dir "${INPUT}/bart-base" \
    --model_name_or_path ${CHECKPOINT}

#cp "${INPUT}/kb.json" ${PROCESSED}

#CUDA_VISIBLE_DEVICES=7 python -m Bart_Program.train \
#    --input_dir ${PROCESSED} \
#    --output_dir "${OUTPUT}/checkpoint" \
#    --save_dir "${OUTPUT}/logs" \
#    --model_name_or_path ${PROCESSED} 

#--num_train_epochs 25

#CUDA_VISIBLE_DEVICES=7 python -m Bart_Program.predict \
#    --input_dir ${PROCESSED} \
#    --save_dir "${OUTPUT}/preds" \
#    --ckpt "${OUTPUT}/checkpoint/checkpoint-147475"