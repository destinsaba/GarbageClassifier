# import packages
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms, models, datasets
from torchvision.models import ResNet18_Weights
from transformers import DistilBertTokenizer, DistilBertModel
import os
from sklearn.metrics import precision_recall_fscore_support

# define data location on cluster
TRAIN_PATH  = "/work/TALC/enel645_2025w/garbage_data/CVPR_2024_dataset_Train"
VAL_PATH    = "/work/TALC/enel645_2025w/garbage_data/CVPR_2024_dataset_Val"
TEST_PATH   = "/work/TALC/enel645_2025w/garbage_data/CVPR_2024_dataset_Test"

# define data location on local machine
#TRAIN_PATH = r"/Users/destinsaba/Documents/MEng/ENEL_645/dataset_group_5/train"
#VAL_PATH = r"/Users/destinsaba/Documents/MEng/ENEL_645/dataset_group_5/val"
#TEST_PATH = r"/Users/destinsaba/Documents/MEng/ENEL_645/dataset_group_5/test"

# load tokenizer and model for the text data
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Define transformations for the images
transform = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
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
        self.tokenizer = tokenizer  # Store tokenizer reference

    def __len__(self):
        return len(self.dataset.samples)

    def __getitem__(self, idx):
        img_path, label = self.dataset.samples[idx]
        
        # Load and transform image
        image = self.dataset.loader(img_path)
        if self.dataset.transform:
            image = self.dataset.transform(image)
        
        # Extract filename text and tokenize
        filename = os.path.splitext(os.path.basename(img_path))[0]  # Safer file parsing
        text_inputs = self.tokenizer(filename, padding="max_length", truncation=True, max_length=32, return_tensors="pt")
        
        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs["attention_mask"]
        if input_ids.dim() > 1:
            input_ids = input_ids.squeeze(0)
        if attention_mask.dim() > 1:
            attention_mask = attention_mask.squeeze(0)
        
        return image, input_ids, attention_mask, label


class ImageTextClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ImageTextClassifier, self).__init__()

        # Image feature extractor
        self.image_extractor = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Freeze the weights of the image extractor
        for param in self.image_extractor.parameters():
            param.requires_grad = False

        # Remove the final layer of the image extractor
        self.image_extractor.fc = nn.Identity()

        # Text feature extractor
        self.text_extractor = DistilBertModel.from_pretrained('distilbert-base-uncased')

        # Freeze first 4 layers of the text extractor, unfreeze last 2 for fine-tuning
        for i, param in enumerate(self.text_extractor.parameters()):
            if i < len(list(self.text_extractor.parameters())) - 2:
                param.requires_grad = False
            else:
                param.requires_grad = True

        # Reduce text feature dimensionality
        self.text_fc = nn.Linear(self.text_extractor.config.hidden_size, 256)

        # Classifier (image output size is 512, text output size is 256)
        self.classifier = nn.Linear(512 + 256, num_classes)

    def forward(self, images, input_ids, attention_mask):
        # Extract image features
        image_features = self.image_extractor(images)

        # Extract text features
        text_outputs = self.text_extractor(input_ids=input_ids, attention_mask=attention_mask)

        # Reduce text feature dimensionality
        text_features = text_outputs.last_hidden_state[:, 0, :]
        text_features = self.text_fc(text_features)

        # Concatenate image and text features
        features = torch.cat((image_features, text_features), dim=1)

        # Classify
        output = self.classifier(features)

        return output
    
def train_model(model, trainloader, valloader, criterion, optimizer, scheduler=None, num_epochs=20, path='./best_model.pth', patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move model to GPU if available
    best_val_loss = np.inf
    epochs_no_improve = 0

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total = 0

            for images, input_ids, attention_mask, labels in (trainloader if phase == "train" else valloader):
                images, input_ids, attention_mask, labels = (
                    images.to(device), input_ids.to(device), attention_mask.to(device), labels.to(device)
                )

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(images, input_ids, attention_mask)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Statistics
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels).item()
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total
            
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            
            # Save history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)

            if phase == "val":
                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(epoch_loss)
                    else:
                        scheduler.step()
                
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    torch.save(model.state_dict(), path)
                    print("Model saved!")
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        return model, history

    print("Training complete")
    return model, history

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for images, input_ids, attention_mask, labels in dataloader:
            images, input_ids, attention_mask, labels = (
                images.to(device), input_ids.to(device), attention_mask.to(device), labels.to(device)
            )
            
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    # Calculate metrics
    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def plot_training_history(history):
    # Plot loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_history.png')

BATCH_SIZE = 32
NUM_CLASSES = 4
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
MODEL_PATH = "./best_model.pth"
PATIENCE = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load datasets
datasets = {
    "train": ImageTextDataset(TRAIN_PATH, transform=transform["train"], tokenizer=tokenizer),
    "val": ImageTextDataset(VAL_PATH, transform=transform["val"], tokenizer=tokenizer),
    "test": ImageTextDataset(TEST_PATH, transform=transform["test"], tokenizer=tokenizer),
}

print(f"Dataset sizes - Train: {len(datasets['train'])}, Val: {len(datasets['val'])}, Test: {len(datasets['test'])}")

# Create dataloaders
dataloaders = {
    "train": DataLoader(datasets["train"], batch_size=BATCH_SIZE, shuffle=True),
    "val": DataLoader(datasets["val"], batch_size=BATCH_SIZE, shuffle=False),
    "test": DataLoader(datasets["test"], batch_size=BATCH_SIZE, shuffle=False),
}

# Instantiate the model
model = ImageTextClassifier(num_classes=NUM_CLASSES)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE, 
    weight_decay=WEIGHT_DECAY
)

# Define scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=3, factor=0.1
)

# Train the model
model, history = train_model(
    model, 
    dataloaders["train"], 
    dataloaders["val"], 
    criterion, 
    optimizer,
    scheduler=scheduler,
    num_epochs=20, 
    path=MODEL_PATH,
    patience=PATIENCE
)

# Plot training history
plot_training_history(history)

# Load the best model and evaluate on the test set
best_model = ImageTextClassifier(num_classes=NUM_CLASSES)
best_model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.to(device)  # Move model to the correct device


# Evaluate on test set
test_metrics = evaluate_model(best_model, dataloaders["test"], criterion, device)

print("\nTest Set Metrics:")
print(f"Loss: {test_metrics['loss']:.4f}")
print(f"Accuracy: {test_metrics['accuracy']:.4f}")
print(f"Precision: {test_metrics['precision']:.4f}")
print(f"Recall: {test_metrics['recall']:.4f}")
print(f"F1 Score: {test_metrics['f1']:.4f}")

# Get class names from the dataset
class_names = datasets['test'].dataset.classes
print(f"\nClasses: {class_names}")

# Confusion matrix and per-class metrics
y_true, y_pred = [], []
with torch.no_grad():
    for images, input_ids, attention_mask, labels in dataloaders["test"]:
        images, input_ids, attention_mask, labels = (
            images.to(device), input_ids.to(device), attention_mask.to(device), labels.to(device)
        )
        outputs = best_model(images, input_ids, attention_mask)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Compute per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)

print("\nPer-class metrics:")
for i, class_name in enumerate(class_names):
    print(f"{class_name}:")
    print(f"  Precision: {precision[i]:.4f}")
    print(f"  Recall: {recall[i]:.4f}")
    print(f"  F1 Score: {f1[i]:.4f}")
    print(f"  Support: {support[i]}")