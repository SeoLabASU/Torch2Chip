PYTHON="/home/mengjian/anaconda3/envs/myenv/bin/python"

if [ ! -d "$DIRECTORY" ]; then
    mkdir ../save
fi

export CUDA_VISIBLE_DEVICES=0

model=vgg7_Q
wbit=4
abit=4
epochs=200
batch_size=128
lr=0.05
loss=cross_entropy
weight_decay=1e-5

dataset="cifar10"
save_path="/home/mengjian/Desktop/ASU_research/Torch2Chip/save/cifar10/vgg7_Q/vgg7_Q_w4_a4_lr0.05_batch128_cross_entropyloss/eval/"
pretrained_model="/home/mengjian/Desktop/ASU_research/Torch2Chip/save/cifar10/vgg7_Q/vgg7_Q_w4_a4_lr0.05_batch128_cross_entropyloss/model_best.pth.tar"
log_file="training.log"

$PYTHON -W ignore ../main.py \
    --save_path ${save_path} \
    --model ${model} \
    --epochs ${epochs} \
    --log_file ${log_file} \
    --lr ${lr} \
    --loss_type ${loss} \
    --wbit ${wbit} \
    --abit ${abit} \
    --dataset ${dataset} \
    --optimizer sgd \
    --fine_tune \
    --resume ${pretrained_model} \
    --evaluate;