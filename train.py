import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import numpy as np
from tqdm import tqdm
import pickle
import multiprocessing

# Dataset sınıfı
class SceneDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Model tanımlama
def create_model(num_classes=6):
    model = EfficientNet.from_pretrained('efficientnet-b0')
    
    # Sadece son birkaç katmanı eğitilebilir yap
    for param in model.parameters():
        param.requires_grad = False
    
    # Son 3 bloğu eğitilebilir yap
    for param in model._blocks[-3:].parameters():
        param.requires_grad = True
    
    # Son katmanı değiştir
    model._fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model._fc.in_features, num_classes)
    )
    
    return model

def main():
    # Basitleştirilmiş veri dönüşümleri
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset ve DataLoader oluşturma
    train_dataset = SceneDataset(root_dir='dataset/seg_train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)  # Batch size azaltıldı

    # Model, loss function ve optimizer tanımlama
    device = torch.device('cpu')
    model = create_model().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Sadece eğitilebilir parametreler için optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=0.001)

    # Eğitim döngüsü
    num_epochs = 3  # Epoch sayısı azaltıldı
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
        
        # Epoch sonunda modeli kaydet
        epoch_loss = running_loss/total
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            model_save = {
                'model_state_dict': model.state_dict(),
                'class_to_idx': train_dataset.class_to_idx,
                'epoch': epoch,
                'loss': best_loss
            }
            with open('scene_classifier.pkl', 'wb') as f:
                pickle.dump(model_save, f)
            print(f"Model saved at epoch {epoch+1} with loss {best_loss:.4f}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main() 