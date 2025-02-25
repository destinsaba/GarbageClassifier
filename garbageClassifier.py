import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import os
import numpy as np
import re
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths
TRAIN_PATH = "/work/TALC/enel645_2025w/garbage_data/CVPR_2024_dataset_Train"
VAL_PATH = "/work/TALC/enel645_2025w/garbage_data/CVPR_2024_dataset_Val"
TEST_PATH = "/work/TALC/enel645_2025w/garbage_data/CVPR_2024_dataset_Test"

# Image Transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Multimodal Dataset Class
class MultimodalGarbageDataset(Dataset):
    def __init__(self, root_dir, tokenizer, max_len, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        self.class_folders = sorted(os.listdir(root_dir))
        self.class_map = {folder: idx for idx, folder in enumerate(self.class_folders)}
        
        self.image_paths = []
        self.texts = []
        self.labels = []
        
        # Collect data
        for class_name in self.class_folders:
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                files = os.listdir(class_path)
                for file_name in files:
                    if file_name.endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_path, file_name)
                        self.image_paths.append(img_path)
                        
                        # Extract text from filename
                        file_name_no_ext, _ = os.path.splitext(file_name)
                        text = file_name_no_ext.replace('_', ' ')
                        text_without_digits = re.sub(r'\d+', '', text)
                        self.texts.append(text_without_digits)
                        
                        # Add label
                        self.labels.append(self.class_map[class_name])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        
        # Image processing
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Text processing
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        
        return {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'text': self.texts[idx],
            'label': torch.tensor(label, dtype=torch.long)
        }

# Image Model Component
class ImageModel(nn.Module):
    def __init__(self, num_classes, freeze_backbone=True):
        super(ImageModel, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Replace the final fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Identity()  # Remove the final layer
        self.fc = nn.Linear(num_features, 512)  # Add a new intermediate layer
        self.activation = nn.ReLU()
        
    def forward(self, x):
        features = self.model(x)
        output = self.fc(features)
        return self.activation(output)

# Text Model Component
class TextModel(nn.Module):
    def __init__(self, freeze_backbone=True):
        super(TextModel, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        if freeze_backbone:
            for param in self.distilbert.parameters():
                param.requires_grad = False
                
        self.fc = nn.Linear(self.distilbert.config.hidden_size, 512)
        self.activation = nn.ReLU()
        
    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0]  # Take the [CLS] token representation
        return self.activation(self.fc(pooled_output))

# Fusion Model
class MultimodalGarbageClassifier(nn.Module):
    def __init__(self, num_classes, freeze_backbones=True):
        super(MultimodalGarbageClassifier, self).__init__()
        self.image_model = ImageModel(num_classes, freeze_backbone=freeze_backbones)
        self.text_model = TextModel(freeze_backbone=freeze_backbones)
        
        # Fusion and classification layers
        self.fusion = nn.Sequential(
            nn.Linear(512 + 512, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, image, input_ids, attention_mask):
        # Get embeddings from individual models
        image_features = self.image_model(image)
        text_features = self.text_model(input_ids, attention_mask)

        # Normalize features
        image_features = nn.functional.normalize(image_features, p=2, dim=1)
        text_features = nn.functional.normalize(text_features, p=2, dim=1)
        
        # Concatenate features
        combined_features = torch.cat((image_features, text_features), dim=1)
        
        # Apply fusion network
        fused_features = self.fusion(combined_features)
        
        # Final classification
        output = self.classifier(fused_features)
        
        return output

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    best_val_loss = float('inf')
    best_model_path = 'best_multimodal_model.pth'
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for i, batch in enumerate(train_loader):
            # Move data to device
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Average training loss for the epoch
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                outputs = model(images, input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Average validation loss and accuracy
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accuracy = 100.0 * correct / total
        
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, '
              f'Val Accuracy: {val_accuracy:.2f}%')
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f'Model saved at epoch {epoch+1}')
    
    # Load best model
    model.load_state_dict(torch.load(best_model_path))
    return model, train_losses, val_losses

# Evaluation function
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in test_loader:
            # Move data to device
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(images, input_ids, attention_mask)
            _, predicted = torch.max(outputs.data, 1)
            
            # Calculate accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Collect predictions and true labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = correct / total
    
    # Create classification report
    class_names = sorted(os.listdir(TRAIN_PATH))
    report = classification_report(all_labels, all_preds, target_names=class_names)
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return accuracy, report, cm, all_preds, all_labels

# Main execution
def main():
    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    max_len = 24
    
    # Create datasets
    train_dataset = MultimodalGarbageDataset(
        TRAIN_PATH, tokenizer, max_len, transform=train_transform)
    val_dataset = MultimodalGarbageDataset(
        VAL_PATH, tokenizer, max_len, transform=test_transform)
    test_dataset = MultimodalGarbageDataset(
        TEST_PATH, tokenizer, max_len, transform=test_transform)
    
    # Create data loaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Initialize model
    num_classes = len(os.listdir(TRAIN_PATH))
    model = MultimodalGarbageClassifier(num_classes, freeze_backbones=True).to(device)
    
    # Training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    num_epochs = 10
    
    # Train model
    print("Starting training...")
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device)
    
    # Evaluate model
    print("\nEvaluating multimodal model...")
    accuracy, report, cm, _, _ = evaluate_model(model, test_loader, device)
    print(f"Multimodal Model Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    class_names = sorted(os.listdir(TRAIN_PATH))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, 'b-', label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curves.png')
    plt.close()
    
    print("\nEvaluation complete. Results saved.")

if __name__ == "__main__":
    main()