#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

# conda activate loftr
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

data_cfg_path="configs/data/megadepth_test_1500.py"
main_cfg_path="configs/loftr/outdoor/buggy_pos_enc/loftr_ds.py"
ckpt_path="D:\LoFTR-master-scalenet\logs\tb_logs\原模型-调参\version_0\checkpoints\epoch=1-auc@5=0.484-auc@10=0.650-auc@20=0.779.ckpt"
#ckpt_path="D:\LoFTR-master-scalenet\logs\\tb_logs\新模型-每层融合\\version_2\checkpoints\epoch=2-auc@5=0.485-auc@10=0.655-auc@20=0.779.ckpt"
dump_dir="dump/loftr_ds_outdoor"
profiler_name="inference"
n_nodes=1  # mannually keep this the same with --nodes
n_gpus_per_node=-1
torch_num_workers=4
batch_size=1  # per gpu

python -u ./test.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --ckpt_path=${ckpt_path} \
    --dump_dir=${dump_dir} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers}\
    --profiler_name=${profiler_name} \
    --benchmark 
    