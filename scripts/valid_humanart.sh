IDX=6,7,8,9

export PYTHONPATH=$PYTHONPATH:./

data_dir=./data/
output_dir=./checkpoints/ckpts/coco

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

output_eval_dir=${output_dir}/humanart_eval
if [ -d ${output_eval_dir} ];then
    echo "dir already exists"
else
    mkdir ${output_eval_dir}
fi

CUDA_VISIBLE_DEVICES=$IDX OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=4 --master_port=25003 \
    utils/valid2d.py \
    --model-name ${output_dir} \
    --question-file ${data_dir}/HumanArt/annotations/validation_humanart.json \
    --image-folder ${data_dir} \
    --output-dir ${output_eval_dir} \
    --conv-format keypoint  2>&1 | tee ${output_eval_dir}/eval.txt