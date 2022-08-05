PYTHON="/home/jmeng15/anaconda3/bin/python"

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
fi

export CUDA_VISIBLE_DEVICES=1

model=vit7
msabit=4
mlpbit=32
epochs=200
batch_size=128
lr=1e-4
loss=ce_smooth

patch=8
num_layer=7
head=12
hidden=384
mlp_hidden=384

dataset="cifar10"
save_path="../save/qViT_msa${msabit}bit_msa${mlp}bit/${dataset}/VIT${num_layer}_${wbit}_lr${lr}_batch${batch_size}_${loss}loss_relu/eval/"
log_file="vit_training.log"
name="vit_training_${dataset}"
pretrained_model="../save/qViT_msa4bit_msabit/cifar10/VIT7__lr1e-4_batch128_ce_smoothloss_relu/model_best.pth.tar"

$PYTHON -W ignore ../main.py \
    --model ${model} \
    --save_path ${save_path} \
    --epochs ${epochs} \
    --log_file ${log_file} \
    --lr  ${lr} \
    --loss_type ${loss} \
    --patch ${patch} \
    --num_layer ${num_layer} \
    --head ${head} \
    --hidden ${hidden} \
    --msabit ${msabit} \
    --msabit ${msabit} \
    --mlp_hidden ${mlp_hidden} \
    --optimizer adam \
    --wandb False \
    --name ${name} \
    --project vit-train \
    --entity jmeng15 \
    --fine_tune \
    --evaluate \
    --resume ${pretrained_model} \
    --ngpu 1;