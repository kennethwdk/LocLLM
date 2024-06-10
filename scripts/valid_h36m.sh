IDX=8,9

export PYTHONPATH=$PYTHONPATH:./

data_dir=./data/
output_dir=./checkpoints/ckpts/h36m

output_eval_dir=${output_dir}/h36m_eval
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
