import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.io import read_image 
from tqdm import tqdm

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.resnet = models.resnet50(pretrained=True)
        
        for param in self.resnet.parameters():
            param.requires_grad = False

        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.authorized_embeddings = {}

    def forward_one(self, x):
        embedding = self.resnet(x)
        return F.normalize(embedding, p=2, dim=1)

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2
    
    def cosine_similarity_loss(self, output1, output2, label):
        cos_sim = F.cosine_similarity(output1, output2)
        loss = (1 - label) * torch.pow(cos_sim, 2) + (label) * torch.pow(torch.clamp(1 - cos_sim, min=0.0), 2)
        return loss.mean()

    def autherise_user(self,image,name):
        self.eval()
        with torch.no_grad():
            embedding= self.forward_one(image)
        self.authorized_embeddings[name] = embedding
    
    def verify_autherisation(self,image,threshold=0.5):
        self.eval()
        with torch.no_grad():
            embedding_new = self.forward_one(image)

        closest_distance = float('inf')
        closest_user = None
        for user, embedding in self.authorized_embeddings.items():
            distance = torch.pairwise_distance(embedding_new, embedding).item()
            if distance < closest_distance:
                closest_distance = distance
                closest_user = user
        if closest_distance < threshold:
            return closest_user,closest_distance
        else:
            return '', closest_distance
        

class SiameseDataset(Dataset):
    def __init__(self, image_folder, transform=None, img_size=(224, 224)):
        self.image_folder = image_folder
        self.transform = transform
        self.img_size = img_size  

        self.classes = os.listdir(image_folder)
        self.image_paths = {class_idx: [] for class_idx, _ in enumerate(self.classes)}

        for class_idx, class_name in enumerate(self.classes):
            class_folder = os.path.join(image_folder, class_name)
            for img_name in os.listdir(class_folder):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):  
                    self.image_paths[class_idx].append(os.path.join(class_folder, img_name))

        self.class_indices = list(self.image_paths.keys())

        self.resize = transforms.Resize(self.img_size)

    def __len__(self):
        return sum(len(imgs) for imgs in self.image_paths.values())

    def __getitem__(self, idx):
        all_images = [(img, lbl) for lbl, imgs in self.image_paths.items() for img in imgs]
        img1_path, label1 = all_images[idx]

        is_positive = np.random.choice([True, False])

        if is_positive:
            img2_path = np.random.choice(self.image_paths[label1])
            label2 = label1
        else:
            negative_label = np.random.choice([lbl for lbl in self.class_indices if lbl != label1])
            img2_path = np.random.choice(self.image_paths[negative_label])
            label2 = negative_label

        img1 = read_image(img1_path).float() / 255.0
        img2 = read_image(img2_path).float() / 255.0

        img1 = self.resize(img1)
        img2 = self.resize(img2)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        label = torch.tensor(1 if label1 == label2 else 0, dtype=torch.float32)

        return img1, img2, label

def transform_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),  
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),  
    ])
    return transform(image)

if __name__ == '__main__':
    dataset = SiameseDataset(image_folder="dataset_train", transform=transform_image)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=16) 

    for img1, img2, label in dataloader:
        print(img1.shape, img2.shape, label.shape)
        break

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0 
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        
        for input1, input2, label in progress_bar:
            input1, input2, label = input1.to(device), input2.to(device), label.to(device)
            
            output1, output2 = model(input1, input2)
            loss = model.cosine_similarity_loss(output1, output2, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {epoch_loss / len(dataloader):.4f}")

    print("Training complete!")

    torch.save(model.state_dict(), 'resnet50_2.pkl')