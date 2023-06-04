INPUT='/data/ljx/data/KQAPro/dataset'
PROCESSED='/home/ljx/new_cache_server32_0411/KQAPro/sparql_processed_test'
CHECKPOINT='/data/ljx/result/probeLLM/sparql/bart-base_test_230601/checkpoint/checkpoint-17472'
OUTPUT='/data/ljx/result/probeLLM/sparql/bart-base_test_230601'

CUDA_VISIBLE_DEVICES=3 python -m Bart_SPARQL.predict \
    --input_dir ${PROCESSED} \
    --save_dir "${OUTPUT}/preds" \
    --ckpt ${CHECKPOINT}
    
    #"${OUTPUT}/checkpoint/checkpoint-67850"