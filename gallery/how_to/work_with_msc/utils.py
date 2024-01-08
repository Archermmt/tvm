# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

""" Utils of using msc examples """

import numpy as np

import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms

import tvm


def get_dataloaders(path, batch_size=1, dataset="cifar10"):
    """Get the data loaders for torch process"""

    if dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        trainset = torchvision.datasets.CIFAR10(
            root=path, train=True, download=True, transform=train_transform
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        testset = torchvision.datasets.CIFAR10(
            root=path, train=False, download=True, transform=test_transform
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        return trainloader, testloader
    raise Exception("Unexpected dataset " + str(dataset))


def run_torch_model(model, dataloader, max_iter=-1, log_step=100, optimizer=None):
    """Run the torch model"""

    device = next(model.parameters()).device
    num_correct, num_datas = 0, 0
    if optimizer:
        model.train()
        criterion = nn.CrossEntropyLoss()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader, 0):
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            cls_idices = torch.argmax(outputs, axis=1)
            labels = labels.to(device)
            num_datas += len(cls_idices)
            num_correct += torch.where(cls_idices == labels, 1, 0).sum()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # gather loss
            running_loss += loss.item()
            if num_datas > 0 and num_datas % log_step == 0:
                print(
                    "[{}/{}] Torch train loss: {}, acc {}".format(
                        i, len(dataloader), running_loss / (i + 1), num_correct / num_datas
                    )
                )
            if max_iter > 0 and num_datas >= max_iter:
                break
    else:
        model.eval()
        for i, (inputs, labels) in enumerate(dataloader, 0):
            with torch.no_grad():
                outputs = model(inputs.to(device))
            cls_idices = torch.argmax(outputs, axis=1)
            labels = labels.to(device)
            num_datas += len(cls_idices)
            num_correct += torch.where(cls_idices == labels, 1, 0).sum()
            if num_datas > 0 and num_datas % log_step == 0:
                print(
                    "[{}/{}] Torch eval acc: {}".format(i, len(dataloader), num_correct / num_datas)
                )
            if max_iter > 0 and num_datas >= max_iter:
                break
    acc = num_correct / num_datas
    return acc.detach().cpu().numpy().tolist()


def run_tvm_model(model, dataloader, max_iter=-1, log_step=100):
    """Run the tvm model"""

    num_correct, num_datas = 0, 0
    for i, (inputs, labels) in enumerate(dataloader, 0):
        for batch_idx, batch_input in enumerate(inputs):
            cls_idx = np.argmax(model(batch_input).asnumpy(), axis=1)
            if cls_idx[0] == labels[batch_idx]:
                num_correct += 1
            num_datas += 1
        if num_datas > 0 and num_datas % log_step == 0:
            print("[{}/{}] Tvm eval acc: {}".format(i, len(dataloader), num_correct / num_datas))
        if max_iter > 0 and num_datas >= max_iter:
            break
    return num_correct / num_datas
