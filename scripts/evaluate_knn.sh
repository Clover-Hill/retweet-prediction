MODEL=/data0/jqcao/models/gpt2-${MODEL_SIZE}
DATASET=/home/jqcao/projects/retweet-prediction/data/feature_dataset

CUDA_VISIBLE_DEVICES=0 python -m \
    evaluate \
    --dataset_name ${DATASET} \
    --do_eval \
    --per_device_eval_batch_size 8 \
    --use_knn \
    --k 20 \
    --sim_threshold 0.8 \
    --viral_threshold 0.5 \
    --index_name "feature_all" \
    --dim 2048 \
    --use_gpu