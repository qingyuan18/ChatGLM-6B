PRE_SEQ_LEN=128
LR=2e-2

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --do_train \
    --train_file $TRAIN_DATASET \
    --validation_file $TEST_DATASET \
    --prompt_column ${PROMPT_COLUMN} \
    --response_column ${RESPONSE_COLUMN}  \
    --overwrite_cache \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --model_output_s3_path ${MODEL_OUTPUT_S3_PATH} \
    --train_simple \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps ${TRAIN_STEPS} \
    --logging_steps 10 \
    --save_steps ${TRAIN_STEPS} \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4

