#!/bin/bash

test_in1k_fid() {
    export LD_LIBRARY_PATH=/u/yzhao/softwares/cudnn-linux-x86_64-9.3.0.75_cuda12-archive/lib/:$LD_LIBRARY_PATH

    # step 1, infer images
    # PYTHONPATH=. ${python_ext} tools/class_conditioned_infer.py \
    PYTHONPATH=. torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 \
    --master_port=12345 \
    tools/class_conditioned_infer.py \
    --cfg ${cfg} \
    --tau ${tau} \
    --top_p ${top_p} \
    --top_k ${top_k} \
    --bs ${bs} \
    --num_per_class 50 \
    --pn ${pn} \
    --model_path ${infinity_model_path} \
    --saln 0 \
    --vae_type ${vae_type} \
    --quantizer_type='MultiScaleLeechQ' \
    --codebook_size=196_560 \
    --vae_path ${vae_path} \
    --add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
    --use_bit_label 0 \
    --use_dit_label 0 \
    --model_type ${model_type} \
    --rope2d_each_sa_layer ${rope2d_each_sa_layer} \
    --rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
    --use_scale_schedule_embedding ${use_scale_schedule_embedding} \
    --cfg ${cfg} \
    --tau ${tau} \
    --checkpoint_type ${checkpoint_type} \
    --text_encoder_ckpt ${text_encoder_ckpt} \
    --text_channels ${text_channels} \
    --apply_spatial_patchify ${apply_spatial_patchify} \
    --cfg_insertion_layer ${cfg_insertion_layer} \
    --out_dir  ${out_dir} \

    # step 2, compute fid
    CUDA_VISIBLE_DEVICES=0 ${python_ext} tools/fid_score.py \
    ${out_dir}/pred \
    /new-pool/Datasets/ILSVRC2012/val_flatten/ | tee ${out_dir}/log.txt 

    cd third_party/guided-diffusion/evaluations/
    CUDA_VISIBLE_DEVICES=1 ${python_ext} evaluator.py VIRTUAL_imagenet256_labeled.npz ../../../${out_dir}/pred.npz
    cd ../../../

}

python_ext=python3
pip_ext="uv pip"

# set arguments for inference
pn=0.06M
model_type=infinity_layer24
use_scale_schedule_embedding=0
use_bit_label=0
checkpoint_type='torch'
infinity_model_path=checkpoints/InfinityCC_L24SQ/generation/infinitycc_24layer_256x256_l24_xl_vf_ep350_cce_zloss_improved_schedule_dion_unsharedaln/ar-ckpt-giter027K-ep349-iter78-last.pth
out_dir_root=output/infinity_layer24_l24_xl_vf_ce_evaluation_improved_schedule
vae_type=24
vae_path=checkpoints/InfinityCC_L24SQ/tokenization/infinity_l24_stage1_xl_vf/model_step_499999.ckpt
cfg="1,1.25,1.5,1.75,2,2.25,2.5"
tau=1
top_p=0.98
top_k=2000,1700,1400,1100,800,500,200
bs=6
rope2d_normalized_by_hw=2
add_lvl_embeding_only_first_block=1
rope2d_each_sa_layer=1
text_encoder_ckpt=weights/flan-t5-xl
text_channels=0
apply_spatial_patchify=0
cfg_insertion_layer=0
sub_fix=cfg${cfg}_tau${tau}_k${top_k}_p${top_p}_cfg_insertion_layer${cfg_insertion_layer}

# IN-1k
out_dir=${out_dir_root}/val_in1k_fid_${sub_fix}
rm -rf ${out_dir}
test_in1k_fid
