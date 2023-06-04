PRED="/data/ljx/result/kqa_pro/bart-base_program_6_7/preds/predict.txt"
GT="/data/ljx/data/KQAPro/test"
CUDA_VISIBLE_DEIVCES=6 python evaluate.py ${GT} ${PRED}