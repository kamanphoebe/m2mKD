import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
from torch.utils.data.dataloader import default_collate


def get_sentence_id(row):
    split, set_id, pair_id, sentence_id = row["identifier"].split("-")
    return sentence_id


def get_image_pair(row, img_root):
    split, set_id, pair_id, sentence_id = row["identifier"].split("-")

    ident = f"{split}-{set_id}-{pair_id}"
    if split == "train":
        img_path_prefix = os.path.join(img_root, str(row["directory"]), ident)
    else:
        img_path_prefix = os.path.join(img_root, ident)

    img1_path = img_path_prefix + "-img0.png"
    img2_path = img_path_prefix + "-img1.png"
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")

    def get_concat_h_resize(
        im1, im2, resample=Image.BICUBIC, resize_big_image=False, dims=(224, 224)
    ):
        new_dims = (dims[0] // 2, dims[1])
        im1 = im1.resize(new_dims, resample=resample)
        im2 = im2.resize(new_dims, resample=resample)
        dst = Image.new("RGB", dims)
        dst.paste(im1, (0, 0))
        dst.paste(im2, (dims[0] // 2, 0))
        return dst

    image = get_concat_h_resize(img1, img2)
    return image, row["sentence"], eval(row["label"])


# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir="hf_cache")

# def nlvr_raw_collate_fn(batch):
#     images, text, labels = zip(*[sample.values() for sample in batch])
#     inputs = processor(text=text, images=images, return_tensors="pt", padding=True,
#                             is_split_into_words=False)

#     return inputs, torch.tensor(labels, dtype=torch.int64)


def nlvr_collate_fn(batch):
    samples, labels = zip(*[sample.values() for sample in batch])
    return torch.stack(samples), torch.tensor(labels, dtype=torch.int64)


def nlvr_collate_patches_fn(batch):
    images, text, labels, attention_masks = zip(*[sample.values() for sample in batch])
    mask = torch.stack(attention_masks)
    return {
        "image": torch.stack(images),
        "text": torch.stack(text),
        "mask": mask,
    }, torch.tensor(labels)


class NLVRRaw(Dataset):
    def __init__(self, dataset_path, train, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.split = "train" if train else "val"
        self.img_root = os.path.join(self.dataset_path, self.split + "_img", self.split)
        self.label_path = os.path.join(dataset_path, self.split + ".json")
        self.labels = pd.read_json(self.label_path, lines=True)
        self.queries = self.labels["query"].unique()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        row = self.labels.iloc[i]
        image, sentence, label = get_image_pair(row, self.img_root)
        # image = self.transform(image)
        return {"image": image, "sentence": sentence, "label": label}


def get_image_pair_no_concat(row, img_root):
    split, set_id, pair_id, sentence_id = row["identifier"].split("-")

    ident = f"{split}-{set_id}-{pair_id}"
    if split == "train":
        img_path_prefix = os.path.join(img_root, str(row["directory"]), ident)
    else:
        img_path_prefix = os.path.join(img_root, ident)

    img1_path = img_path_prefix + "-img0.png"
    img2_path = img_path_prefix + "-img1.png"
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")
    return img1, img2, row["sentence"], eval(row["label"])


class NLVRRawNoConcat(Dataset):
    def __init__(self, dataset_path, train, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.split = "train" if train else "val"
        self.img_root = os.path.join(self.dataset_path, self.split + "_img", self.split)
        self.label_path = os.path.join(dataset_path, self.split + ".json")
        self.labels = pd.read_json(self.label_path, lines=True)
        self.queries = self.labels["query"].unique()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        row = self.labels.iloc[i]
        image1, image2, sentence, label = get_image_pair_no_concat(row, self.img_root)
        image1 = self.transform(image1)
        image2 = self.transform(image2)
        return {
            "lhs_image": image1,
            "rhs_image": image2,
            "sentence": sentence,
        }, torch.tensor(label, dtype=torch.long)

    @classmethod
    def collate_fn(cls, batch):
        lhs_images = [sample[0]["lhs_image"] for sample in batch]
        lhs_images = default_collate(lhs_images)
        rhs_images = [sample[0]["rhs_image"] for sample in batch]
        rhs_images = default_collate(rhs_images)
        text = [sample[0]["sentence"] for sample in batch]

        samples = dict(lhs_images=lhs_images, rhs_images=rhs_images, text=text)
        targets = torch.tensor([sample[1] for sample in batch], dtype=torch.int64)
        return samples, targets


class NLVRPreprocessed(Dataset):
    def __init__(self, dataset_path, train, transform=None):
        self.dataset_path = dataset_path
        self.split = "train" if train else "val"
        self.img_root = os.path.join(self.dataset_path, self.split + "_img", self.split)
        self.label_path = os.path.join(dataset_path, self.split + ".json")
        self.labels = pd.read_json(self.label_path, lines=True)
        self.queries = self.labels["query"].unique()
        self.samples = torch.load(
            os.path.join(dataset_path, "nlvr2-" + self.split + "-preprocessed.pt")
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        row = self.labels.iloc[i]
        sample = self.samples[i]
        return {"sample": sample, "label": eval(row["label"])}


class NLVRPreprocessedPatches(Dataset):
    def __init__(self, dataset_path, train, transform=None):
        self.dataset_path = dataset_path
        self.split = "train" if train else "val"
        self.img_root = os.path.join(self.dataset_path, self.split + "_img", self.split)
        self.label_path = os.path.join(dataset_path, self.split + ".json")
        dataset = torch.load(
            os.path.join(dataset_path, "nlvr2-" + self.split + "-preprocessed.pt")
        )
        self.lhs_patches = torch.cat(dataset["patches_lhs"], dim=0)
        del dataset["patches_lhs"]
        self.rhs_patches = torch.cat(dataset["patches_rhs"], dim=0)
        del dataset["patches_rhs"]
        self.text = torch.cat(dataset["text"], dim=0)
        del dataset["text"]
        self.labels = torch.cat(dataset["labels"], dim=0).to(torch.int64)
        del dataset["labels"]
        self.attention_masks = torch.cat(dataset["attention_masks"], dim=0).to(
            torch.bool
        )
        del dataset["attention_masks"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        image = torch.stack([self.lhs_patches[i], self.rhs_patches[i]], dim=0)
        label = self.labels[i]
        mask = torch.cat(
            [
                torch.ones(image.shape[0] * image.shape[1], dtype=torch.bool),
                self.attention_masks[i],
            ]
        )
        return {
            "image": image,
            "text": self.text[i],
            "label": label,
            "attention_mask": mask,
        }
