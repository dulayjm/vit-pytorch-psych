# Justin Dulay
# this is just a scratch file for working with ViT+PP

from __future__ import print_function

import glob
from itertools import chain
import os
import random
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

from torch.utils.data.sampler import SubsetRandomSampler

from vit_pytorch.efficient import ViT

# Dataset class bloat
from torch.utils.data import Dataset
import os
import pandas as pd
import random
from skimage import io

from PIL import Image, ImageFilter

from torchvision.utils import save_image

# reaction time psychophysical loss
def _softmax(x):
    exp_x = torch.exp(x)
    sum_x = torch.sum(exp_x, dim=1, keepdim=True)

    return exp_x/sum_x

def _log_softmax(x):
    return torch.log(_softmax(x))

def RtPsychCrossEntropyLoss(outputs, targets, psych):
    num_examples = targets.shape[0]
    batch_size = outputs.shape[0]

    # converting reaction time to penalty
    # 10002 is close to the max penalty time seen in the data
    for idx in range(len(psych)):   
        psych[idx] = abs(10002 - psych[idx])

    # adding penalty to each of the output logits 
    for i in range(len(outputs)):
        val = psych[i] / 300
        if np.isnan(val.cpu()):
            val = 0 
            
        outputs[i] += val 

    outputs = _log_softmax(outputs)
    outputs = outputs[range(batch_size), targets]

    return - torch.sum(outputs) / num_examples

class OmniglotReactionTimeDataset(Dataset):
    """
    Dataset for omniglot + reaction time data
    Dasaset Structure:
    label1, label2, real_file, generated_file, reaction time
    ...
    args:
    - path: string - path to dataset (should be a csv file)
    - transforms: torchvision.transforms - transforms on the data
    """

    def __init__(self, data_file, transforms=None):
        self.raw_data = pd.read_csv(data_file)
        self.transform = transforms

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        label1 = int(self.raw_data.iloc[idx, 0])
        label2 = int(self.raw_data.iloc[idx, 1])

        im1name = self.raw_data.iloc[idx, 2]
        image1 = Image.open(im1name)
        im2name = self.raw_data.iloc[idx, 3]
        image2 = Image.open(im2name)
        
        rt = self.raw_data.iloc[idx, 4]
        sigma_or_accuracy = self.raw_data.iloc[idx, 5]
        
        # if you wanted to, you could perturb one of the images. 
        # our final experiments did not do this, though. only some of them 
        # image1 = image1.filter(ImageFilter.GaussianBlur(radius = sigma_or_accuracy))

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        sample = {'label1': label1, 'label2': label2, 'image1': image1,
                                            'image2': image2, 'rt': rt, 'acc': sigma_or_accuracy}

        return sample

# Training settings
batch_size = 64
epochs = 20
lr = 3e-5
gamma = 0.7
seed = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

# change for when on a GPU 
device = 'cpu'

os.makedirs('data', exist_ok=True)
train_dir = 'data/train'
test_dir = 'data/test'

# TODO: load in our psych data here, like from omniglot, etc 



# train_list, valid_list = train_test_split(train_list, 
#                                           test_size=0.2,
#                                           stratify=labels,
#                                           random_state=seed)

# print(f"Train Data: {len(train_list)}")
# print(f"Validation Data: {len(valid_list)}")
# print(f"Test Data: {len(test_list)}")

# data transforms and loader 
train_transform = transforms.Compose([
                # transforms.RandomCrop(64, padding=0),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ])

# Datasets
dataset = OmniglotReactionTimeDataset('small_dataset.csv', 
            transforms=train_transform)

test_split = .2
shuffle_dataset = True

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(test_split * dataset_size))

if shuffle_dataset:
    np.random.seed(2)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                        sampler=train_sampler)
# when did we ever downsample these images?

# test loader not utilzed in the train file 
# _ = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                                 sampler=valid_sampler)
# import torchvision
# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#     plt.imsave('sanity.jpg')

# dataiter = iter(train_loader)
# images, labels = dataiter.next()

# show images
# imshow(torchvision.utils.make_grid(images))
# print labels
# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

# Image Augmentation
# train_transforms = transforms.Compose(
#     [
#         transforms.Resize((224, 224)),
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#     ]
# )

# val_transforms = transforms.Compose(
#     [
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#     ]
# )


# test_transforms = transforms.Compose(
#     [
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#     ]
# )

# Data Loaders

# Model

# you can change the specific type of transformer here, too
efficient_transformer = Linformer(
    dim=128,
    seq_len=5,  # 7x7 patches + 1 cls-token
    depth=12,
    heads=8,
    k=64
)

model = ViT(
    dim=128,
    image_size=64,
    patch_size=16,
    num_classes=100,
    transformer=efficient_transformer,
    channels=3,
).to(device)

# Training Definitions
# loss function
# TODO: see if we can use the other loss, or if we should just run w the other stuff
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

# Train loop 
# for epoch in range(epochs):
#     epoch_loss = 0
#     epoch_accuracy = 0

#     for data, label in tqdm(train_loader):
#         data = data.to(device)
#         label = label.to(device)

#         output = model(data)
#         loss = criterion(output, label)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         acc = (output.argmax(dim=1) == label).float().mean()
#         epoch_accuracy += acc / len(train_loader)
#         epoch_loss += loss / len(train_loader)

#     # with torch.no_grad():
#     #     epoch_val_accuracy = 0
#     #     epoch_val_loss = 0
#     #     for data, label in valid_loader:
#     #         data = data.to(device)
#     #         label = label.to(device)

#     #         val_output = model(data)
#     #         val_loss = criterion(val_output, label)

#     #         acc = (val_output.argmax(dim=1) == label).float().mean()
#     #         epoch_val_accuracy += acc / len(valid_loader)
#     #         epoch_val_loss += val_loss / len(valid_loader)

# new train loop
import time
accuracies = []
losses = []
exp_time = time.time()
optimizer = optim.Adam(model.parameters(), lr=lr)


for epoch in range(2):
    running_loss = 0.0
    correct = 0.0
    total = 0.0

    for idx, sample in enumerate(train_loader):
        image1 = sample['image1']
        image2 = sample['image2']

        label1 = sample['label1']
        label2 = sample['label2']

        # if args.loss_fn == 'psych-acc':
        #     psych = sample['acc']
        # else: 
        psych = sample['rt']

        # concatenate the batched images
        inputs = torch.cat([image1, image2], dim=0).to(device)
        labels = torch.cat([label1, label2], dim=0).to(device)

        # apply psychophysical annotations to correct images
        psych_tensor = torch.zeros(len(labels))
        j = 0 
        for i in range(len(psych_tensor)):
            if i % 2 == 0: 
                psych_tensor[i] = psych[j]
                j += 1
            else: 
                psych_tensor[i] = psych_tensor[i-1]
        psych_tensor = psych_tensor.to(device)

        outputs = model(inputs).to(device)        
        loss = RtPsychCrossEntropyLoss(outputs, labels, psych_tensor).to(device)

        # update weights and back propogate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # calculate accuracy per class
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total

    print(f'epoch {epoch} accuracy: {accuracy:.2f}%')
    print(f'running loss: {train_loss:.4f}')

    # if args.use_neptune: 
    #     neptune.log_metric('train_loss', train_loss)
    #     neptune.log_metric('accuracy', accuracy)

    accuracies.append(accuracy)
    losses.append(train_loss)

    print(f'{time.time() - exp_time:.2f} seconds')


