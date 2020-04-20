# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch
import time
from tqdm import tqdm_notebook as tqdm
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage import io
import torch.optim as optim
from torchvision import datasets, transforms, models
import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.optim import lr_scheduler
 # %%
device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
model = models.resnet50(pretrained=True)
print(model)

# %%
test_dataframe = pd.read_csv('/home/awall03/Datafile/smallerdataset.csv')
X = test_dataframe['image_id']
y = test_dataframe['diagnosis'].astype(int)
X = ['/home/awall03/Datafile/data/preprocessed_train/' + i + '.png' for i in X if ".png" not in i]
image_count = len(X)
classes = ('Normal','Mild','Moderate','Severe','Proliferative DR')

# %%
type(y)
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, random_state=42)
# %%
# %%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

# %%

class DataLoader():
    def __init__(self,imagepaths, imageclass):
            
        self.len = len(imagepaths)
        self.x_data = imagepaths
        self.y_data = imageclass
    def __getitem__(self, index):
        image = Image.open(self.x_data[index])
        image =  image.resize((256, 256), resample=Image.BILINEAR)
        label = torch.tensor(self.y_data.iloc[index])
        return { 'image': transforms.ToTensor()(image),
                'labels': label 
                }
    def __len__(self):
        return self.len


# %%

training_data = DataLoader(X_train,y_train)
test_data = DataLoader(X_test,y_test)
training_data.__getitem__(0)
# %%
data_loader = torch.utils.data.DataLoader(training_data, batch_size=128, shuffle=False, num_workers=0)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True, num_workers=4)

# %%
model = torchvision.models.resnet101(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(2048, 1)

model = model.to(device)


plist = [
         {'params': model.layer4.parameters(), 'lr': 1e-4, 'weight': 0.001},
         {'params': model.fc.parameters(), 'lr': 1e-3}
         ]

optimizer = optim.Adam(plist, lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10)
# %%
# for i in data_loader:
# # %%
# since = time.time()
# criterion = nn.MSELoss()
# num_epochs = 15
# for epoch in range(num_epochs):
#     print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#     print('-' * 10)
#     scheduler.step()
#     model.train()
#     running_loss = 0.0
#     counter = 0
#     for d in data_loader:
#         inputs = d["image"]
#         labels = d["labels"].view(-1, 1)
#         inputs = inputs.to(device, dtype=torch.float)
#         labels = labels.to(device, dtype=torch.float)
#         optimizer.zero_grad()
#         with torch.set_grad_enabled(True):
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#         running_loss += loss.item() * inputs.size(0)
#         counter += 1
#         epoch_loss = running_loss / len(data_loader)
#     print('Training Loss: {:.4f}'.format(epoch_loss))

# time_elapsed = time.time() - since
# print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
# torch.save(model.state_dict(), "model.bin")

# %%
since = time.time()
criterion = nn.MSELoss()
num_epochs = 15
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    scheduler.step()
    model.train()
    running_loss = 0.0
    tk0 = tqdm(data_loader, total=int(len(data_loader)))
    counter = 0
    for bi, d in enumerate(tk0):
        inputs = d["image"]
        labels = d["labels"].view(-1, 1)
        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        counter += 1
        tk0.set_postfix(loss=(running_loss / (counter * data_loader.batch_size)))
    epoch_loss = running_loss / len(data_loader)
    print('Training Loss: {:.4f}'.format(epoch_loss))

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
torch.save(model.state_dict(), "model.bin")

# %%
