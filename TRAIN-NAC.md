# Training NAC using m2mKD

If you would like to quickly try out m2mKD on NACs, we have provided the checkpoints of the teacher modules or the student modules for Tiny-ImageNet, which can be downloaded from [here](https://huggingface.co/kamanphoebe/m2mKD/nac_tinyimnet_students). You can either use the checkpoints of the teacher modules to skip the preparation phase, or use the checkpoints of the student modules to skip both the preparation and the m2mKD phase.

- **Datasets**: 
    - Preparation/m2mKD: [ImageNet-1k](https://www.image-net.org/download.php)
    - End-to-end: [Tiny-ImageNet](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet), ImageNet-1k, [Tiny-ImageNet-R](https://zenodo.org/records/6653675) (optional), [ImageNet-R](https://github.com/hendrycks/imagenet-r) (optional) 
    - Few-shot: [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
- **Teacher model:** DeiT-Huge (32 layers)
- **Student model:** NAC (10 layers)
- **Num of modules:** 10

## Preparation phase

The preparation phase involves training a teacher model from scratch.

The following commands used in this phase are similar to the [instructions](https://github.com/LeapLabTHU/Deep-Incubation/blob/master/TRAINING.md) provided by Deep Incubation. You may check it out for more details.

```bash
# 1. Meta model pretraining (PT)
# Prepare a meta model with 10 layers.
torchrun --standalone --nnodes=1 --nproc_per_node=8 main.py \
--phase PT --model vit_huge_patch14_224 \
--divided_depths 1 1 1 1 1 1 1 1 1 1 --output_dir ./log_dir/PT_huge \
--batch_size 256 --epochs 300 --drop-path 0 --use_amp 1

# 2. Modular training (MT)
# Incubate teacher modules using the meta model trained in the previous step. 
# Each module can be incubated in parallel.
# Note that the first and last sub-modules contain 4 layers, and the remaining sub-modules comprise 3 layers.
torchrun --standalone --nnodes=1 --nproc_per_node=8 main.py \
--phase MT --model vit_huge_patch14_224  --meta_ckpt_path ./log_dir/PT_huge/finished_checkpoint.pth \
--idx 0 --divided_depths 4 1 1 1 1 1 1 1 1 1 --output_dir ./log_dir/MT_huge_0 \
# --idx 1 --divided_depths 1 3 1 1 1 1 1 1 1 1 --output_dir ./log_dir/MT_huge_1 \
# --idx 2 --divided_depths 1 1 3 1 1 1 1 1 1 1 --output_dir ./log_dir/MT_huge_2 \
# ...
# --idx 8 --divided_depths 1 1 1 1 1 1 1 1 3 1 --output_dir ./log_dir/MT_huge_8 \
# --idx 9 --divided_depths 1 1 1 1 1 1 1 1 1 4 --output_dir ./log_dir/MT_huge_9 \
--batch_size 128 --update_freq 2 --epochs 100 --drop-path 0.1 --use_amp 1

# 3. Assembly & Fine-tuning (FT)
torchrun --standalone --nnodes=1 --nproc_per_node=8 main.py \
--phase FT --model vit_huge_patch14_224 --incubation_models ./log_dir/MT_huge_*/finished_checkpoint.pth \
--divided_depths 4 3 3 3 3 3 3 3 4 --output_dir ./log_dir/FT_huge  \
--batch_size 128 --update_freq 1 --epochs 100 --warmup-epochs 0 --clip-grad 1 --drop-path 0.1 --use_amp 1
```

## m2mKD phase

After obtaining the teacher modules, you can run m2mKD for each teaching pair. The pairs can be processed in parallel.

```bash
# Note that the idx of the last pair should be -1. 
# If you want to train student modules for ImageNet, use "--student_cfg_path ./configs/nac_student_config_imnet.yml" instead.
torchrun --standalone --nnodes=1 --nproc_per_node=8 main.py \
--phase m2mKD --model nac --stitch_dim 1024 --student_cfg_path ./configs/nac_student_tinyimnet_config.yml \
--idx 0 --teacher_ckpt_path ./log_dir/FT_huge/vit_huge_patch14_224_0.pth --output_dir ./log_dir/m2mKD_nac_0 \
# --idx 1 --teacher_ckpt_path ./log_dir/FT_huge/vit_huge_patch14_224_1.pth --output_dir ./log_dir/m2mKD_nac_1 \
# --idx 2 --teacher_ckpt_path ./log_dir/FT_huge/vit_huge_patch14_224_2.pth --output_dir ./log_dir/m2mKD_nac_2 \
# ...
# --idx 8 --teacher_ckpt_path ./log_dir/FT_huge/vit_huge_patch14_224_8.pth --output_dir ./log_dir/m2mKD_nac_8 \
# --idx -1 --teacher_ckpt_path ./log_dir/FT_huge/vit_huge_patch14_224_9.pth --output_dir ./log_dir/m2mKD_nac_9 \
--meta vit_huge_patch14_224 --meta_divided_depths 1 1 1 1 1 1 1 1 1 1 --meta_ckpt_path ./log_dir/PT_huge/finished_checkpoint.pth \
--batch_size 128 --update_freq 2 --distillation_type soft --epochs 100  --use_amp 1
```

## End-to-end traing phase

The final phase is to load the learned parameters into the student model and perform end-to-end training.

```bash
# Move the student modules to another directory and rename them.
bash copy_nacstudent.sh

# Copy the configuration file you would like to use.
mkdir -p ./nacs/experiments/nac_erdos_tinyimnet/Configurations
cp ./configs/nac_erdos_tinyimnet_config.yml ./nacs/experiments/nac_erdos_tinyimnet/Configurations

cd nacs
# Use wandb to record the training process.
wandb login
wandb init 
wandb enabled
# Train the student model in end-to-end manner.
torchrun --standalone --nnodes=1 --nproc_per_node=8 train_supervised.py experiments/nac_erdos_tinyimnet
```

## Few-shot adaptation

The command below can be used to evaluate the few-shot adaptation performance on CIFAR-100. In our experiments, we maintain the number of validation samples to be three times the number of training samples (i.e. $\text{val-sample} = 3 \times \text{n-shot}$). These settings can be modified in `nac_scale_fewshot_config.yml`.

```bash
mkdir -p ./nacs/experiments/nac_scale_fewshot/Configurations
cp ./configs/nac_scale_fewshot_config.yml ./nacs/experiments/nac_scale_fewshot/Configurations
cd nacs & torchrun --standalone --nnodes=1 --nproc_per_node=8 train_supervised.py experiments/nac_scale_fewshot
```