# import packages
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms, models, datasets
import torchvision.transforms.functional as F
from torchvision.models import resnet18
from transformers import DistilBertTokenizer, DistilBertModel
from PIL import Image
import os

# define data location on cluster
# TRAIN_PATH  = "/work/TALC/enel645_2025w/garbage_data/CVPR_2024_dataset_Train"
# VAL_PATH    = "/work/TALC/enel645_2025w/garbage_data/CVPR_2024_dataset_Val"
# TEST_PATH   = "/work/TALC/enel645_2025w/garbage_data/CVPR_2024_dataset_Test"

# define data location on local machine
TRAIN_PATH = r"/Users/destinsaba/Documents/MEng/ENEL_645/dataset_group_5/train"
VAL_PATH = r"/Users/destinsaba/Documents/MEng/ENEL_645/dataset_group_5/val"
TEST_PATH = r"/Users/destinsaba/Documents/MEng/ENEL_645/dataset_group_5/test"

# load tokenizer and model for the text data
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Define transformations for the images
transform = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

# class to load the dataset, assumes that image filenames are the relevant text
class ImageTextDataset(Dataset):
    def __init__(self, root_dir, transform=None, tokenizer=None):
        self.dataset = datasets.ImageFolder(root=root_dir, transform=transform)
        self.tokenizer = tokenizer
        self.samples = [(img_path, label) for img_path, label in self.dataset.samples]
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = self.dataset.loader(img_path)
        if self.dataset.transform:
            image = self.dataset.transform(image)
        
        # Extract filename text and tokenize
        filename = os.path.basename(img_path).split('.')[0]  # Remove extension
        text_inputs = self.tokenizer(filename, padding='max_length', truncation=True, max_length=32, return_tensors="pt")
        
        return image, text_inputs["input_ids"].squeeze(0), text_inputs["attention_mask"].squeeze(0), label

# Load datasets
datasets = {
    "train": ImageTextDataset(TRAIN_PATH, transform=transform["train"], tokenizer=tokenizer),
    "val": ImageTextDataset(VAL_PATH, transform=transform["val"], tokenizer=tokenizer),
    "test": ImageTextDataset(TEST_PATH, transform=transform["test"], tokenizer=tokenizer),
}



# The following code is temporary and just tests the dataset class / helps understand the tokenizer / preprocessing

# Print count of samples per class for train, val, and test datasets
for split in ["train", "val", "test"]:
    labels = []
    for _, _, _, label in datasets[split]:
        labels.append(label)
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"{split} dataset class counts:")
    for label, count in zip(unique_labels, counts):
        print(f"Class {label}: {count} samples")

# Display a sample image and its corresponding tokenized input IDs
sample_image, sample_input_ids, sample_attention_mask, sample_label = datasets["train"][0]
print(f"Image shape: {sample_image.shape}")

# These come from the DistilBert tokenizer and are not interpretable
print(f"Tokenized input IDs: {sample_input_ids}")
print(f"Attention Mask: {sample_attention_mask}")

decoded_text = tokenizer.decode(sample_input_ids, skip_special_tokens=True)
print(decoded_text)  


print(f"Label: {sample_label}")

# The image looks weird since it is normalized (PIL is used to display the tensor)
plt.imshow(F.to_pil_image(sample_image))
plt.axis("off")
plt.show()