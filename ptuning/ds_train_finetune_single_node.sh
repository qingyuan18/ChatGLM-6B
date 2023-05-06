
LR=1e-4
SM_MASTER="${SM_MASTER}"
SM_MASTER_ADDR="${SM_MASTER_ADDR}"
CURRENT_HOST="${SM_CURRENT_HOST}"
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
export NCCL_SOCKET_IFNAME="eth0"

deepspeed --num_gpus=$NUM_GPUS  --master_port $MASTER_PORT main.py \
    --deepspeed deepspeed.json \
    --do_train \
    --train_file $TRAIN_DATASET \
    --test_file $TEST_DATASET \
    --prompt_column ${PROMPT_COLUMN} \
    --response_column ${RESPONSE_COLUMN} \
    --overwrite_cache \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --model_output_s3_path ${MODEL_OUTPUT_S3_PATH} \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --max_steps ${TRAIN_STEPS} \
    --logging_steps 10 \
    --save_steps ${TRAIN_STEPS} \
    --learning_rate $LR \
    --fp16

