import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import models, transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score

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
        if x.dim() == 3: 
            x = x.unsqueeze(0)  
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork()
state_dict = torch.load("resnet50_2.pkl")
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
print('')

class testDataset(Dataset):
    def __init__(self, root_dir, authorized_users, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.authorized_users = authorized_users
        self.data = []
        self.authorized_image_name = [['dummy_path','dummy_user']]

        for user in os.listdir(root_dir):
            user_dir = os.path.join(root_dir, user)
            if os.path.isdir(user_dir):
                for img_name in os.listdir(user_dir):
                    img_path = os.path.join(user_dir, img_name)
                    if user in authorized_users:
                        label = 1
                        if user not in [row[1] for row in self.authorized_image_name]:
                            image = cv2.imread(img_path)
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            if self.transform:
                                image = self.transform(image)
                            self.authorized_image_name.append([image,user])
                    else:
                        label = 0
                    self.data.append((img_path, label))  
        self.authorized_image_name.pop(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, label
    
def evaluate_with_authorized_users(model, dataset, threshold=0.1):
    data = [(img, label) for img, label in dataset]

    model.eval()
    with torch.no_grad():
        for image, user in dataset.authorized_image_name:
            image = image.to(device) 
            model.autherise_user(image, user)

    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for image, true_label in tqdm(data, desc="Evaluating Dataset"):
            image = image.to(device)
            predicted_user, _ = model.verify_autherisation(image, threshold)
            true_labels.append(true_label)
            predicted_labels.append(1 if predicted_user else 0)

    accuracy = accuracy_score(true_labels, predicted_labels)
    return accuracy

def transform_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),  
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),  
    ])
    return transform(image)

dataset_path = "dataset_2"
authorized_users = ['class1' 'class2','class3','class4','class5','deepak','deepanshu','deepesh','gaurav','gopal','jewel','kailash','manoj','saksham']

transform = lambda img: transform_image(img).to(device) 
test_dataset = testDataset(dataset_path, authorized_users, transform=transform)

accuracy= evaluate_with_authorized_users(model, test_dataset, threshold=0.5)
print(f"Model Accuracy: {accuracy * 100:.2f}%")