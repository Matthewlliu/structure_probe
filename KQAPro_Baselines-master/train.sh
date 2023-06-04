INPUT='/data/ljx/result/probeLLM/kopl/lf2nl_kopl_text-davinci-003_2023-03-27_94376'
PROCESSED='/data/ljx/data/KQAPro/davinci-003_lf2nl'
CHECKPOINT='/data/ljx/cpt/bart-base'
#CHECKPOINT='/data/ljx/result/para_model/bart-base-stage-5epochs_2022-06-07/checkpoints/checkpoint-18750'

OUTPUT='/data/ljx/result/kqa_pro/bart-base_davinci-003_230329'

CUDA_VISIBLE_DEVICES=2 python -m Bart_Program.train \
    --input_dir ${PROCESSED} \
    --output_dir "${OUTPUT}/checkpoint" \
    --save_dir "${OUTPUT}/logs" \
    --model_name_or_path ${PROCESSED} \
    --num_train_epochs 10 \
    --batch_size 32 \
    --learning_rate 1e-4