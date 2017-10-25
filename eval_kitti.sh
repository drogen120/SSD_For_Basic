EVAL_DIR=./logs/
CHECKPOINT_PATH=./logs/model.ckpt-129319
DATASET_DIR=./tf_records/
python eval_ssd_network.py \
        --eval_dir=${EVAL_DIR} \
        --dataset_dir=${DATASET_DIR} \
        --dataset_name=kitti \
        --dataset_split_name=train \
        --model_name=ssd_300_vgg \
        --checkpoint_path=${CHECKPOINT_PATH} \
        --batch_size=1
