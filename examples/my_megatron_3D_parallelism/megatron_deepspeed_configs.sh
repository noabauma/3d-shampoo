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

NLAYERS=1
NHIDDEN=1024
BATCHSIZE=$(jq '.train_micro_batch_size_per_gpu' ${config_json})           # actually micro batch-size for pipeline modules


GAS=16 # Gradient Accumulation Steps

#ZeRO Configs
stage=$(jq '.zero_optimization.stage' ${config_json})
reduce_scatter=true
contigious_gradients=false
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
        --data-path $DATA_PATH \
        --vocab-file $VOCAB_PATH \
        --merge-file $MERGE_PATH \
        --distributed-backend nccl \
        --lr 1.5e-4 \
        --seed 42 
"
  
 deepspeed_options=" \
                --deepspeed \
                --deepspeed_config ${config_json} \
                --zero-stage ${stage} \
                --zero-reduce-bucket-size ${rbs} \
                --zero-allgather-bucket-size ${agbs} 
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

# nvprof --profile-from-start off --profile-child-processes --metrics flop_count_sp 
run_cmd="python megatron_3D.py $@ ${full_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x