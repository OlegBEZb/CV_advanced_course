import argparse
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from data_utils import IntelImageDataset
import models
from eval import eval_clf

IMG_WIDTH, IMG_HEIGHT = 150, 150


@torch.no_grad()
def get_preds(model, loader):
    all_preds = torch.tensor([]).to(device)
    all_labels = torch.tensor([])
    for batch in loader:
        images, labels = batch
        images = images.to(device)
        # delete me
        images = images.float()
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


def train_nn(model, train_loader, val_dataloader, optimizer, es, epochs=5):
    print('>>> Training Start >>>')
    for epoch in tqdm(range(epochs), total=epochs):
        total_loss = 0
        total_correct = 0
        for batch in tqdm(train_loader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            # delete me
            images = images.float()
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir', type=str, default='./', required=True)
    parser.add_argument('--test_data_dir', type=str, default='./', required=True)
    parser.add_argument('--val_size', type=float, default=0.2, required=True)
    parser.add_argument('--model', type=str, default='cnn', required=False)
    parser.add_argument('--load_on_fly', action='store_true')
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    train_data_dir = args.train_data_dir
    test_data_dir = args.test_data_dir
    val_size = args.val_size
    load_on_fly = args.load_on_fly
    epochs = args.epochs

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    le = LabelEncoder()

    train_dataset = IntelImageDataset(img_dir=train_data_dir,
                                      transform=transforms.Resize([IMG_WIDTH, IMG_HEIGHT]),
                                      target_transform=le,
                                      mode='train',
                                      load_on_fly=load_on_fly)

    val_dataset = IntelImageDataset(img_dir=train_data_dir,
                                    transform=transforms.Resize([IMG_WIDTH, IMG_HEIGHT]),
                                    target_transform=le,
                                    mode='val',
                                    load_on_fly=load_on_fly)

    test_dataset = IntelImageDataset(img_dir=test_data_dir,
                                     transform=transforms.Resize([IMG_WIDTH, IMG_HEIGHT]),
                                     target_transform=train_dataset.target_transform,
                                     mode='test',
                                     load_on_fly=load_on_fly)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    if args.model == 'cnn':
        model = models.CNN_network(num_classes=len(train_dataset.classes))
    elif args.model == 'cnn_mlp':
        model = models.CNN_MLP_network(num_classes=len(train_dataset.classes))
    else:
        import torchvision

        model = torchvision.models.wide_resnet50_2(pretrained=True)
        for param in model.parameters():
            param.required_grad = False
        num_ftrt = model.fc.in_features
        model.fc = nn.Linear(num_ftrt, len(train_dataset.classes))

    model = model.to(device)

    early_stopper = EarlyStopper(patience=3)
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    train_nn(model, train_dataloader, val_dataloader, optimizer, early_stopper, epochs=epochs)
