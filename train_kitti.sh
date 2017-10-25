DATASET_DIR=./tf_records
TRAIN_DIR=./logs/
CHECKPOINT_PATH=./checkpoints/ssd_300_vgg.ckpt
python train_ssd_network.py \
        --train_dir=${TRAIN_DIR} \
        --dataset_dir=${DATASET_DIR} \
        --dataset_name=kitti \
        --dataset_split_name=train \
        --model_name=ssd_300_vgg \
        --checkpoint_path=${CHECKPOINT_PATH} \
        --weight_decay=0.0005 \
        --optimizer=adam \
        --learning_rate=0.001 \
        --batch_size=12
