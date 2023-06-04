INPUT='/data/ljx/data/KQAPro/dataset'
PROCESSED='/data/ljx/data/KQAPro/davinci-003_lf2nl'
#CHECKPOINT='/data/ljx/cpt/new_download_bart_kqa'
CHECKPOINT='/data/ljx/result/kqa_pro/bart-base_davinci-003_230329/checkpoint/checkpoint-29500'

OUTPUT='/data/ljx/result/kqa_pro/bart-base_davinci-003_230329'
CUDA_VISIBLE_DEVICES=7 python -m Bart_Program.predict \
    --input_dir ${PROCESSED} \
    --save_dir "${OUTPUT}/preds" \
    --ckpt ${CHECKPOINT}
    
    #"${OUTPUT}/checkpoint/checkpoint-67850"