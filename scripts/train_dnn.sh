DATASET=/fs-computility/plm/shared/jqcao/projects/retweet-prediction/data/feature_dataset
NUM_EPOCH=50
LR="1e-3"

ACCELERATE_CONFIG=./accelerate_config/train-8-card.yaml
# ACCELERATE_CONFIG=./accelerate_config/eval.yaml
# MODEL=/fs-computility/plm/shared/jqcao/models/Qwen3/Qwen3-Embedding-0.6B
OUTPUT_DIR=/fs-computility/plm/shared/jqcao/projects/retweet-prediction/checkpoint/dnn-epoch-${NUM_EPOCH}-${LR}

WANDB_PROJECT="retweet-prediction" accelerate launch \
    --config_file ${ACCELERATE_CONFIG} \
    -m \
    train \
    --dataset_path ${DATASET} \
    --output_dir ${OUTPUT_DIR} \
    --learning_rate ${LR} \
    --lr_scheduler_type linear \
    --gradient_accumulation_steps 4 \
    --per_device_train_batch_size 1024 \
    --per_device_eval_batch_size 1024 \
    --num_train_epochs ${NUM_EPOCH} \
    --model_type feature_only \
    --seed 42 \
    --logging_steps 100 \
    --checkpointing_steps "epoch" \
    --logging_steps 1 \
    --report_to wandb \
    --with_tracking \
    --project_name retweet-prediction \
    --run_name dnn-epoch-${NUM_EPOCH}-${LR}