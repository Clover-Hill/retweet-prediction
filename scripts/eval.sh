DATASET=/fs-computility/plm/shared/jqcao/projects/retweet-prediction/data/preprocessed_dataset

# ACCELERATE_CONFIG=./accelerate_config/eval.yaml
ACCELERATE_CONFIG=./accelerate_config/train-8-card.yaml
CLASSIFICATION_MODEL=/fs-computility/plm/shared/jqcao/projects/retweet-prediction/checkpoint/all-classification-epoch-50/epoch_5
REGRESSION_MODEL=/fs-computility/plm/shared/jqcao/projects/retweet-prediction/checkpoint/multi-regression-epoch-50-1e-4/epoch_16
OUTPUT_DIR=./result/threshold-86

accelerate launch \
    --config_file ${ACCELERATE_CONFIG} \
    -m \
    eval \
    --classification_model_path ${CLASSIFICATION_MODEL} \
    --regression_model_path ${REGRESSION_MODEL} \
    --dataset_name ${DATASET} \
    --output_dir ${OUTPUT_DIR} \
    --per_device_eval_batch_size 32 \
    --do_test \
    --viral_threshold 0.86 \
    --report_to none