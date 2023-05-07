#!/bin/bash
WORKING_DIR=/opt/ml/code/
SM_WORKING_DIR=/opt/ml/model

#The related information about multi-nodes cluster.
NNODES="$NODE_NUMBER"
NODE_RANK="$NODE_INDEX"
LR=1e-4
SM_MASTER="${SM_MASTER}"
SM_MASTER_ADDR="${SM_MASTER_ADDR}"
CURRENT_HOST="${SM_CURRENT_HOST}"
MASTER_PORT="23456"
export NCCL_SOCKET_IFNAME="eth0"

#Configure the distributed arguments for torch.distributed.launch.
GPUS_PER_NODE="$SM_NUM_GPUS"
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

SAVE_PATH="${SM_WORKING_DIR}/results"
LOG_FILE="${SAVE_PATH}/log.txt"

#Set the path of your deepspeed config file.
DS_CONFIG="${WORKING_DIR}/deepspeed.json"
OPTS=""" --deepspeed ${WORKING_DIR}/deepspeed.json
    --do_train 
    --train_mutipl 
    --train_file $TRAIN_DATASET 
    --test_file $TEST_DATASET 
    --prompt_column ${PROMPT_COLUMN} 
    --response_column ${RESPONSE_COLUMN} 
    --overwrite_cache 
    --model_name_or_path ${MODEL_NAME_OR_PATH} 
    --output_dir ${OUTPUT_DIR} 
    --model_output_s3_path ${MODEL_OUTPUT_S3_PATH} 
    --overwrite_output_dir 
    --max_source_length 64 
    --max_target_length 64 
    --per_device_train_batch_size 1 
    --per_device_eval_batch_size 4 
    --gradient_accumulation_steps 1 
    --predict_with_generate 
    --max_steps ${TRAIN_STEPS} 
    --logging_steps 10 
    --save_steps ${TRAIN_STEPS} 
    --learning_rate $LR 
    --fp16"""
CMD="python -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${WORKING_DIR}/main.py ${OPTS}"
echo ${CMD}
mkdir -p ${SAVE_PATH}
${CMD} 2>&1 | tee ${SAVE_PATH}/train_log
