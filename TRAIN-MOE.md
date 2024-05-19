# Training V-MoE-Base using m2mKD

- **Datasets**: 
    - Preparation/m2mKD/End-to-end: ImageNet-1k 
    - Few-shot: CIFAR-100, CUB-2011
- **Teacher model:** DeiT-Large (24 layers)
- **Student model:** V-MoE-Base (12 MoE layers, 8 experts)
- **Num of modules:** 4

## Preparation phase

The preparation phase involves training a teacher model from scratch. If you prefer to use the released checkpoint of Deep Incubation as the teacher model, you can follow the steps below to obtain the teacher modules and proceed directly to the m2mKD phase:
1. Download the [checkpoint](https://huggingface.co/nzl-thu/Deep-Incubation).
2. `cd m2mKD ; mkdir ./log_dir/FT_large/ ; mv /PATH/TO/CKPT ./log_dir/FT_large/finished_checkpoint.pth`
3. Run `python split_teacher.py` to split the teacher model into modules.

The following commands used in this phase are similar to the [instructions](https://github.com/LeapLabTHU/Deep-Incubation/blob/master/TRAINING.md) provided by Deep Incubation. You may check it out for more details.

```bash
# 1. Meta model pretraining (PT)
# Prepare a meta model with 4 layers.
torchrun --standalone --nnodes=1 --nproc_per_node=8 main.py \
--phase PT --model vit_large_patch16_224 \
--divided_depths 1 1 1 1 --output_dir ./log_dir/PT_large \
--batch_size 256 --epochs 300 --drop-path 0 --use_amp 1

# 2. Modular training (MT)
# Incubate teacher modules using the meta model trained in the previous step. 
# Each module can be incubated in parallel.
torchrun --standalone --nnodes=1 --nproc_per_node=8 main.py \
--phase MT --model vit_large_patch16_224  --meta_ckpt_path ./log_dir/PT_large/finished_checkpoint.pth \
--idx 0 --divided_depths 6 1 1 1 --output_dir ./log_dir/MT_large_0 \
# --idx 1 --divided_depths 1 6 1 1 --output_dir ./log_dir/MT_large_1 \
# --idx 2 --divided_depths 1 1 6 1 --output_dir ./log_dir/MT_large_2 \
# --idx 3 --divided_depths 1 1 1 6 --output_dir ./log_dir/MT_large_3 \
--batch_size 128 --update_freq 2 --epochs 100 --drop-path 0.1 --use_amp 1

# 3. Assembly & Fine-tuning (FT)
torchrun --standalone --nnodes=1 --nproc_per_node=8 main.py \
--phase FT --model vit_large_patch16_224 --incubation_models ./log_dir/MT_large_*/finished_checkpoint.pth \
--divided_depths 6 6 6 6 --output_dir ./log_dir/FT_large  \
--batch_size 128 --update_freq 1 --epochs 100 --warmup-epochs 0 --clip-grad 1 --drop-path 0.1 --use_amp 1
```

## m2mKD phase

After obtaining the teacher modules, you can run m2mKD for each teaching pair. The pairs can be processed in parallel.

```bash
# Note that the idx of the last pair should be -1. 
torchrun --standalone --nnodes=1 --nproc_per_node=8 main.py \
--phase m2mKD --model vmoe_student_base --divided_depths 3 --moe_pos 0 1 2 --num_expert 8 --stitch_dim 1024  \
--idx 0 --post_stitch_pos 2 --output_dir ./log_dir/m2mKD_vmoe_0 --teacher_ckpt_path ./log_dir/FT_large/vit_large_patch16_224_0.pth \
# --idx 1 --pre_stitch_pos 0 --post_stitch_pos 2 --output_dir ./log_dir/m2mKD_vmoe_1 --teacher_ckpt_path ./log_dir/FT_large/vit_large_patch16_224_1.pth \
# --idx 2 --pre_stitch_pos 0 --post_stitch_pos 2 --output_dir ./log_dir/m2mKD_vmoe_2 --teacher_ckpt_path ./log_dir/FT_large/vit_large_patch16_224_2.pth \
# --idx -1 --pre_stitch_pos 0 --output_dir ./log_dir/m2mKD_vmoe_3 --teacher_ckpt_path ./log_dir/FT_large/vit_large_patch16_224_3.pth \
--meta vit_large_patch16_224 --meta_divided_depths 1 1 1 1 --meta_ckpt_path ./pretrained_models/PT_large.pth \
--batch_size 128 --update_freq 2 --distillation-type soft --epochs 100  --use_amp 1 
```

## End-to-end traing phase

The final phase is to load the learned parameters into the student model and perform end-to-end training.

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 main.py \
--phase FT --model vmoe_base --incubation_models ./log_dir/m2mKD_vmoe_*/finished_checkpoint.pth \
--divided_depths 3 3 3 3 --output_dir ./log_dir/FT_vmoe \
--moe_pos 0 1 2 3 4 5 6 7 8 9 10 11 --num_expert 8 \
# Uncomment the following line if you want to preserve the stitch layers in the final model.
# --pre_stitch_pos 3 6 9 --post_stitch_pos 2 5 8 --stitch_dim 1024 \
--batch_size 128 --update_freq 1 --epochs 100 --warmup-epochs 0 --clip-grad 1 --drop-path 0.1 --use_amp 1
```

## Few-shot adaptation

The command below can be used to evaluate the few-shot adaptation performance on CIFAR-100 or CUB-2011. In our experiments, we maintain the number of validation samples to be three times the number of training samples (i.e. $\text{val-sample} = 3 \times \text{n-shot}$).

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 main.py \
--phase FS --model vmoe_base --finetune ./log_dir/FT_vmoe/finished_checkpoint.pth \
--divided_depths 3 3 3 3 --output_dir ./log_dir/FS_vmoe \
--moe_pos 0 1 2 3 4 5 6 7 8 9 10 11 --num_expert 8 \
--data-set CIFAR100-FS --k_way 8 --n_shot 5 --val_sample 15 \
# --data-set CUB-FS --k_way 8 --n_shot 5 --val_sample 15 \
--update_freq 1 --epochs 100 --lr 0.002 --seed 1 
```

## Downstream task: COCO object detection & instance segmentation

We follow identical training receipt of [ViTDet](https://github.com/open-mmlab/mmdetection/tree/main/projects/ViTDet) to fine-tune the V-MoE-Base model pretrained by m2mKD for the downstream tasks. To prepare the data, you have to first download the COCO dataset (2017 version) and put the files under the `detectron2/projects/ViTDet/datasets/coco` directory. 

```bash
cd detectron2/projects/ViTDet
../../tools/lazyconfig_train_net.py --config-file configs/COCO/mask_rcnn_vmoe_b_m2mkd.py --num-gpus 8
```