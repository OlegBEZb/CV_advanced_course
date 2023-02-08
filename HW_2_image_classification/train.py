import argparse
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from data_utils import IntelImageDataset
import models
from eval import eval_clf

IMG_WIDTH, IMG_HEIGHT = 150, 150


@torch.no_grad()
def get_preds(model, loader, device):
    all_preds = torch.tensor([]).to(device)
    all_labels = torch.tensor([])
    for batch in loader:
        images, labels = batch
        images = images.to(device)
        preds = model(images)
        all_preds = torch.cat((all_preds, preds), dim=0)
        all_labels = torch.cat((all_labels, labels), dim=0)

    all_labels = all_labels.type(torch.LongTensor).to(device)
    return all_preds, all_labels


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            print('Early stopping became worse. Now counter is', self.counter)
            if self.counter >= self.patience:
                return True
        return False


def train_nn(model, train_loader, val_dataloader, optimizer, es, device, epochs=5):
    print('>>> Training Start >>>')
    for epoch in tqdm(range(epochs), total=epochs):
        total_loss = 0
        total_correct = 0
        for batch in tqdm(train_loader, mininterval=10):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            # https://discuss.pytorch.org/t/f-cross-entropy-vs-torch-nn-cross-entropy-loss/25505
            # print('train types', predictions, labels)
            loss = F.cross_entropy(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss.item()
            total_correct = total_correct + predictions.argmax(dim=1).eq(labels).sum().item()

        val_preds, val_labels = get_preds(model, val_dataloader)
        # print('val types', val_preds, val_labels)
        val_loss = F.cross_entropy(val_preds, val_labels).item()
        f1_val = eval_clf(y_test=val_labels.cpu().numpy(), y_pred=val_preds.argmax(dim=1).cpu().numpy())
        print('epoch:', epoch, "train_loss:", total_loss, 'val_loss:', val_loss, 'val f1:', f1_val)

        if es.early_stop(val_loss):
            break

    print('>>> Training Complete >>>')


def get_dataloaders(train_data_dir, test_data_dir, val_size, transform=transforms.Resize([IMG_WIDTH, IMG_HEIGHT]),
                    transform_test=transforms.Resize([IMG_WIDTH, IMG_HEIGHT])):
    if dataset == 'custom':
        le = LabelEncoder()

        train_dataset = IntelImageDataset(img_dir=train_data_dir,
                                          transform=transform,
                                          target_transform=le,
                                          mode='train',
                                          load_on_fly=load_on_fly,
                                          reduced_num=reduced_num)

        val_dataset = IntelImageDataset(img_dir=train_data_dir,
                                        transform=transform,
                                        target_transform=le,
                                        mode='val',
                                        load_on_fly=load_on_fly,
                                        reduced_num=reduced_num)

        test_dataset = IntelImageDataset(img_dir=test_data_dir,
                                         transform=transform_test,
                                         target_transform=train_dataset.target_transform,
                                         mode='test',
                                         load_on_fly=load_on_fly,
                                         reduced_num=reduced_num)

        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    elif dataset == 'default':
        # ImageFloder function uses for make dataset by passing dir adderess as an argument
        # works a bit slower than when all the images are loaded in init. but works great on the fly
        # TODO: check why native is faster

        # ImageFloder function uses for make dataset by passing dir adderess as an argument
        train_dataset = datasets.ImageFolder(root=train_data_dir, transform=transform)
        test_dataset = datasets.ImageFolder(root=test_data_dir, transform=transform_test)

        # Split data into train and validation set
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(val_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=2, sampler=train_sampler)
        val_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=2, sampler=valid_sampler)
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    else:
        raise

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir', type=str, default='./', required=True)
    parser.add_argument('--test_data_dir', type=str, default='./', required=True)
    parser.add_argument('--dataset', type=str, default='default', required=True)
    parser.add_argument('--val_size', type=float, default=0.2, required=True)
    parser.add_argument('--model', type=str, default='cnn', required=False)
    parser.add_argument('--load_on_fly', action='store_true')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--reduced_num', type=int, required=False)
    args = parser.parse_args()
    print('received args\n', args)
    train_data_dir = args.train_data_dir
    test_data_dir = args.test_data_dir
    dataset = args.dataset
    val_size = args.val_size
    load_on_fly = args.load_on_fly
    epochs = args.epochs
    reduced_num = args.reduced_num

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(train_data_dir=train_data_dir, test_data_dir=test_data_dir, val_size=val_size)

    if args.model == 'cnn':
        model = models.CNN_network(num_classes=6)
    elif args.model == 'cnn_mlp':
        model = models.CNN_MLP_network(num_classes=6)
    else:
        model = torchvision.models.wide_resnet50_2(pretrained=True)
        for param in model.parameters():
            param.required_grad = False
        num_ftrt = model.fc.in_features
        model.fc = nn.Linear(num_ftrt, 6)

    model = model.to(device)

    early_stopper = EarlyStopper(patience=3)
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    train_nn(model, train_dataloader, val_dataloader, optimizer, early_stopper, epochs=epochs)
