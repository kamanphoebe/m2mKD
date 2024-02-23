import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import (
    CLIPProcessor,
    CLIPModel,
    LxmertTokenizer,
    LxmertModel,
    BertTokenizer,
)

from neural_compilers.data.nlvr import NLVRRawNoConcat
from neural_compilers.model.external.ALBEF.model_nlvr import ALBEF

# Setup configurations
on_mila = os.environ.get("SCRATCH") is not None
make_patch_dataset = True
model_name = "albef"  # "clip", lxmert
num_workers = 0  # 30
batch_size = 16  # 128
device = "cuda"
scratch_dir = (
    os.environ.get("SCRATCH") + "/" if os.environ.get("SCRATCH") is not None else ""
)

if on_mila:
    dataset_root = f"{scratch_dir}nlvr2-lxmert/"
    cache_dir = "$SCRATCH/hf_cache/"
else:
    cache_dir = "hf_cache/"
    dataset_root = (
        "/home/XXXX-6/code/neural-compilers/neural_compilers/data/assets/nlvr2/"
    )

# Make the model
if model_name == "lxmert":
    backbone = LxmertModel.from_pretrained(
        "unc-nlp/lxmert-base-uncased", cache_dir=cache_dir
    ).to("cuda")
    processor = LxmertTokenizer.from_pretrained(
        "unc-nlp/lxmert-base-uncased", cache_dir=cache_dir
    )

    def collate_fn(batch):
        lhs_images, rhs_images, text, labels = zip(
            *[sample.values() for sample in batch]
        )
        text = processor(
            text=text, return_tensors="pt", padding="max_length", max_length=64
        )
        return (
            {
                "images": lhs_images,
                "input_ids": text["input_ids"],
                "attention_mask": text["attention_mask"],
            },
            {
                "images": rhs_images,
                "input_ids": text["input_ids"],
                "attention_mask": text["attention_mask"],
            },
            torch.tensor(labels, dtype=torch.int64),
        )

elif model_name == "clip":
    backbone = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32", cache_dir=cache_dir
    )
    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32", cache_dir="hf_cache"
    )

    def collate_fn(batch):
        lhs_images, rhs_images, text, labels = zip(
            *[sample.values() for sample in batch]
        )
        lhs_inputs = processor(
            text=text,
            images=lhs_images,
            return_tensors="pt",
            padding="max_length",
            max_length=64,
            is_split_into_words=False,
        )
        rhs_inputs = processor(
            text=text,
            images=rhs_images,
            return_tensors="pt",
            padding="max_length",
            max_length=64,
            is_split_into_words=False,
        )
        return lhs_inputs, rhs_inputs, torch.tensor(labels, dtype=torch.int64)

elif model_name == "albef":
    import yaml

    config = yaml.load(
        open("neural_compilers/model/ALBEF/nlvr.yml", "r"), Loader=yaml.Loader
    )
    text_encoder = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(text_encoder)
    #### Model ####
    print("Creating model")
    model = ALBEF(config=config, text_encoder=text_encoder, tokenizer=tokenizer)

    def collate_fn(batch):
        lhs_images, rhs_images, text, labels = zip(
            *[sample.values() for sample in batch]
        )
        lhs_images = [np.array(img) for img in lhs_images]
        text = tokenizer(
            text=text, return_tensors="pt", padding="max_length", max_length=64
        )
        return (
            {
                "images": lhs_images,
                "input_ids": text["input_ids"],
                "attention_mask": text["attention_mask"],
            },
            {
                "images": rhs_images,
                "input_ids": text["input_ids"],
                "attention_mask": text["attention_mask"],
            },
            torch.tensor(labels, dtype=torch.int64),
        )


model = model.to(device)
breakpoint()

# setup dataset
train_dataset = NLVRRawNoConcat(dataset_root, train=True, transform=None)
val_dataset = NLVRRawNoConcat(dataset_root, train=False, transform=None)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    collate_fn=collate_fn,
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    collate_fn=collate_fn,
)

if make_patch_dataset:
    patches_lhs = []
    patches_rhs = []
    text = []
    attention_masks = []
    labels = []

    dataloaders = [
        ("nlvr2-val-lxmert-preprocessed.pt", val_dataloader),
        ("nlvr2-train-lxmert-preprocessed.pt", train_dataloader),
    ]
    for fname, data_loader in dataloaders:
        with torch.inference_mode():
            breakpoint()
            for lhs_samples, rhs_samples, targets in tqdm(data_loader):
                lhs_samples = {k: v.to("cuda") for k, v in lhs_samples.items()}

                x = backbone(**lhs_samples)
                patches_lhs.append(x.vision_model_output.last_hidden_state.cpu())

                rhs_samples = {k: v.to("cuda") for k, v in rhs_samples.items()}
                x = backbone(**rhs_samples)
                patches_rhs.append(x.vision_model_output.last_hidden_state.cpu())

                text.append(x.text_model_output.last_hidden_state.cpu())
                attention_masks.append(rhs_samples["attention_mask"].cpu().bool())
                labels.append(targets.cpu().to(torch.int32))
            torch.save(
                {
                    "patches_lhs": patches_lhs,
                    "patches_rhs": patches_rhs,
                    "text": text,
                    "attention_masks": attention_masks,
                    "labels": labels,
                },
                os.path.join(dataset_root, fname),
            )

else:
    train_out_fname = "nlvr2-train-preprocessed.pt"
    val_out_fname = "nlvr2-val-preprocessed.pt"
    train_samples = []
    for samples, targets in tqdm(train_dataloader):
        samples = {k: v.to("cuda:0", non_blocking=True) for k, v in samples.items()}
        x = backbone(**samples)
        x = torch.cat(
            [x.text_model_output.pooler_output, x.vision_model_output.pooler_output],
            dim=-1,
        )
        train_samples.append(x.cpu().detach().numpy())
    train_samples = torch.cat([torch.tensor(sample) for sample in train_samples], dim=0)
    torch.save(train_samples, train_out_fname)

    val_samples = []
    for samples, targets in tqdm(val_dataloader):
        samples = {k: v.to("cuda:1", non_blocking=True) for k, v in samples.items()}
        x = backbone(**samples)
        x = torch.cat(
            [x.text_model_output.pooler_output, x.vision_model_output.pooler_output],
            dim=-1,
        )
        val_samples.append(x.cpu().detach().numpy())
    val_samples = torch.cat([torch.tensor(sample) for sample in val_samples], dim=0)
    torch.save(val_samples, val_out_fname)
