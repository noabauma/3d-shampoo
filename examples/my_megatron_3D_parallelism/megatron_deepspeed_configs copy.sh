#! /bin/bash

DATA_PATH=megatron/data/webtext/webtext_text_document
VOCAB_PATH=megatron/data/gpt2-vocab.json
MERGE_PATH=megatron/data/gpt2-merges.txt

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
config_json="ds_config.json"

# Megatron Model Parallelism
mp_size=1
# DeepSpeed Pipeline parallelism
pp_size=1

NLAYERS=24
NHIDDEN=1024
BATCHSIZE=4 #micro batch-size


GAS=16

#ZeRO Configs
stage=0
reduce_scatter=true
contigious_gradients=true
rbs=50000000
agbs=5000000000


gpt_options=" \
        --model-parallel-size ${mp_size} \
        --pipe-parallel-size ${pp_size} \
        --num-layers $NLAYERS \
        --hidden-size $NHIDDEN \
        --num-attention-heads 16 \
        --seq-length 1024 \
        --max-position-embeddings 1024 \
        --batch-size $BATCHSIZE \
        --gas $GAS \
        --train-iters 1 \
        --lr-decay-iters 1 \
        --data-path $DATA_PATH \
        --vocab-file $VOCAB_PATH \
        --merge-file $MERGE_PATH \
        --data-impl mmap \
        --split 949,50,1 \
        --distributed-backend nccl \
        --lr 1.5e-4 \
        --lr-decay-style cosine \
        --min-lr 1.0e-5 \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --warmup 0.01 \
        --checkpoint-activations \
        --log-interval 1 \
        --save-interval 500 \
        --eval-interval 100 \
        --eval-iters 10 
"
  
 deepspeed_options=" \
                --deepspeed \
                --deepspeed_config ${config_json} \
                --zero-stage ${stage} \
                --zero-reduce-bucket-size ${rbs} \
                --zero-allgather-bucket-size ${agbs} \
                --local_rank=0
                "

if [ "${contigious_gradients}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
                --zero-contigious-gradients"
fi

if [ "${reduce_scatter}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
                --zero-reduce-scatter"
fi


full_options="${gpt_options} ${deepspeed_options}"

run_cmd="python megatron_3D.py $@ ${full_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
