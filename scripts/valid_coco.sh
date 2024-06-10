IDX=6,7,8,9

export PYTHONPATH=$PYTHONPATH:./

data_dir=./data/
output_dir=./checkpoints/ckpts/coco

output_eval_dir=${output_dir}/coco_eval
if [ -d ${output_eval_dir} ];then
    echo "dir already exists"
else
    mkdir ${output_eval_dir}
fi

CUDA_VISIBLE_DEVICES=$IDX OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=4 --master_port=25003 \
    utils/valid2d.py \
    --model-name ${output_dir} \
    --question-file ${data_dir}/coco/annotations/coco_val.json \
    --image-folder ${data_dir}/coco/val2017 \
    --output-dir ${output_eval_dir} \
    --conv-format keypoint  2>&1 | tee ${output_eval_dir}/eval.txt