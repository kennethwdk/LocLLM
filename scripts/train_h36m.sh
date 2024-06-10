IDX=8,9

export PYTHONPATH=$PYTHONPATH:./

data_dir=./data/
output_dir=./checkpoints/ckpts/h36mmpii-2gpu

if [ -d ${output_dir} ];then
    echo "dir already exists"
else
    mkdir ${output_dir}
fi

if [ -d ${output_dir}/src ];then
    echo "src dir already exists"
else
    echo "save codes to src"
    mkdir ${output_dir}/src
    cp -r datasets ${output_dir}/src
    cp -r models ${output_dir}/src
    cp -r utils ${output_dir}/src
    cp -r scripts ${output_dir}/src
fi

CUDA_VISIBLE_DEVICES=$IDX OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=2 --master_port=25002 \
    utils/train3d.py \
    --dataset_name Human36MDataset@MPII3DDataset \
    --model_name_or_path ./checkpoints/model_weights/vicuna-7b-v1.5 \
    --llama_path ./checkpoints/model_weights/vicuna-7b-v1.5 \
    --data_path ./data/h36m/annotations/Sample_trainmin_train_Human36M_protocol_2.json@./data/mpii/annotations/annot_mpii.json \
    --image_folder ./data/h36m/images@./data/mpii/images/ \
    --dino_path ./checkpoints/model_weights/dinov2_vitl14_pretrain.pth \
    --conv_format keypoint \
    --data_augmentation False \
    --tune_mm_mlp_adapter True \
    --freeze_llm False \
    --lora_llm_enable True \
    --freeze_vit False \
    --lora_vision_enable True \
    --bf16 True \
    --output_dir ${output_dir} \
    --num_train_epochs 6 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500000 \
    --save_total_limit 1 \
    --learning_rate 5e-4 \
    --weight_decay 0.05 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --dataloader_num_workers 4 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --report_to tensorboard \
    2>&1 | tee ${output_dir}/log.txt

output_eval_dir=${output_dir}/eval
if [ -d ${output_eval_dir} ];then
    echo "dir already exists"
else
    mkdir ${output_eval_dir}
fi

CUDA_VISIBLE_DEVICES=$IDX OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=2 --master_port=25002 \
    utils/valid3d.py \
    --model-name ${output_dir} \
    --question-file ./data/h36m/Sample_64_test_Human36M_protocol_2.json \
    --image-folder ./data/h36m/images \
    --output-dir ${output_eval_dir} \
    --conv-format keypoint  2>&1 | tee ${output_eval_dir}/eval.txt
