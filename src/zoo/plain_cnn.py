#!/usr/bin/env python3
"""
Plain CNN (no skip connections) for CIFAR-10.
Used instead of ResNet-20 because layer-by-layer weight matching alignment
is straightforward without residual connection constraints.

Architecture: VGG-style
  Conv(3,32) -> BN -> ReLU -> Conv(32,32) -> BN -> ReLU -> MaxPool
  Conv(32,64) -> BN -> ReLU -> Conv(64,64) -> BN -> ReLU -> MaxPool
  Conv(64,128) -> BN -> ReLU -> Conv(128,128) -> BN -> ReLU -> MaxPool(->4x4)
  FC(128*4*4, 256) -> ReLU -> FC(256, 10)

~1.2M params, reaches ~92% on CIFAR-10.
"""

import argparse
import itertools
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms


class Cutout:
    def __init__(self, size=16):
        self.size = size
    def __call__(self, img):
        h, w = img.shape[1], img.shape[2]
        y, x = np.random.randint(h), np.random.randint(w)
        y1, y2 = max(0, y - self.size//2), min(h, y + self.size//2)
        x1, x2 = max(0, x - self.size//2), min(w, x + self.size//2)
        img[:, y1:y2, x1:x2] = 0.0
        return img


class PlainCNN(nn.Module):
    """VGG-style plain CNN without skip connections."""
    def __init__(self, num_classes=10):
        super().__init__()
        # Block 1: 32 channels
        self.conv1a = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.bn1b = nn.BatchNorm2d(32)

        # Block 2: 64 channels
        self.conv2a = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.bn2b = nn.BatchNorm2d(64)

        # Block 3: 128 channels
        self.conv3a = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn3a = nn.BatchNorm2d(128)
        self.conv3b = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.bn3b = nn.BatchNorm2d(128)

        # Classifier
        self.fc1 = nn.Linear(128 * 4 * 4, 256, bias=True)
        self.fc2 = nn.Linear(256, num_classes, bias=True)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Alignment spec for PlainCNN: each layer's output channels can be freely permuted
def get_perm_spec_plaincnn():
    """Return permutation groups for PlainCNN.

    Since there are no skip connections, each layer's output channels
    are the next layer's input channels, and we can permute freely.
    """
    groups = [
        # conv1a output (32) -> conv1b input
        {'size': 32,
         'output': [('conv1a.weight', 0), ('bn1a.weight', 0), ('bn1a.bias', 0),
                    ('bn1a.running_mean', 0), ('bn1a.running_var', 0)],
         'input': [('conv1b.weight', 1)]},
        # conv1b output (32) -> conv2a input
        {'size': 32,
         'output': [('conv1b.weight', 0), ('bn1b.weight', 0), ('bn1b.bias', 0),
                    ('bn1b.running_mean', 0), ('bn1b.running_var', 0)],
         'input': [('conv2a.weight', 1)]},
        # conv2a output (64) -> conv2b input
        {'size': 64,
         'output': [('conv2a.weight', 0), ('bn2a.weight', 0), ('bn2a.bias', 0),
                    ('bn2a.running_mean', 0), ('bn2a.running_var', 0)],
         'input': [('conv2b.weight', 1)]},
        # conv2b output (64) -> conv3a input
        {'size': 64,
         'output': [('conv2b.weight', 0), ('bn2b.weight', 0), ('bn2b.bias', 0),
                    ('bn2b.running_mean', 0), ('bn2b.running_var', 0)],
         'input': [('conv3a.weight', 1)]},
        # conv3a output (128) -> conv3b input
        {'size': 128,
         'output': [('conv3a.weight', 0), ('bn3a.weight', 0), ('bn3a.bias', 0),
                    ('bn3a.running_mean', 0), ('bn3a.running_var', 0)],
         'input': [('conv3b.weight', 1)]},
        # conv3b output (128) -> fc1 input (reshape: 128*4*4 = 2048)
        # For fc1 input, we need to permute groups of 16 (=4*4) columns
        {'size': 128,
         'output': [('conv3b.weight', 0), ('bn3b.weight', 0), ('bn3b.bias', 0),
                    ('bn3b.running_mean', 0), ('bn3b.running_var', 0)],
         'input_fc': [('fc1.weight', 1, 16)]},  # special: fc, axis 1, spatial=4*4=16
        # fc1 output (256) -> fc2 input
        {'size': 256,
         'output': [('fc1.weight', 0), ('fc1.bias', 0)],
         'input': [('fc2.weight', 1)]},
    ]
    return groups


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_dataloaders(aug="basic", batch_size=128, num_workers=4, data_dir="./data"):
    test_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    train_list = [
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    if aug == "cutout":
        train_list.append(Cutout(size=16))
    train_transform = transforms.Compose(train_list)
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += inputs.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += inputs.size(0)
    return 100.0 * correct / total


def run_training(seed, lr, wd, aug, epochs, gpu, save_dir, data_dir="./data"):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"PlainCNN: seed={seed} lr={lr} wd={wd} aug={aug} device={device}")
    print(f"{'='*60}")

    train_loader, test_loader = get_dataloaders(aug=aug, batch_size=128, data_dir=data_dir)
    model = PlainCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        if epoch % 10 == 0 or epoch == epochs:
            test_acc = evaluate(model, test_loader, device)
            print(f"  Epoch {epoch:3d}/{epochs}  loss={train_loss:.4f}  "
                  f"train={train_acc:.1f}%  test={test_acc:.2f}%  [{time.time()-t0:.0f}s]")
            best_acc = max(best_acc, test_acc)

    final_acc = evaluate(model, test_loader, device)
    print(f"  >> Final: {final_acc:.2f}%  (best: {best_acc:.2f}%)")

    os.makedirs(save_dir, exist_ok=True)
    ckpt_name = f"plaincnn_s{seed}_lr{lr}_wd{wd}_aug{aug}.pt"
    ckpt_path = os.path.join(save_dir, ckpt_name)
    torch.save({
        "model_state_dict": model.state_dict(),
        "test_acc": final_acc,
        "hparams": {"seed": seed, "lr": lr, "wd": wd, "aug": aug, "epochs": epochs},
    }, ckpt_path)
    print(f"  >> Saved: {ckpt_path}")
    return final_acc


ALL_SEEDS = list(range(10))
ALL_LRS = [0.01, 0.05, 0.1]
ALL_WDS = [1e-4, 5e-4]
ALL_AUGS = ["basic", "cutout"]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, nargs="+", default=[0])
    p.add_argument("--lr", type=float, nargs="+", default=[0.1])
    p.add_argument("--wd", type=float, nargs="+", default=[1e-4])
    p.add_argument("--aug", type=str, nargs="+", default=["basic"])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--save-dir", type=str, default="/home/dzheng/MergeabilityComplex/checkpoints_plain")
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--train-all", action="store_true")
    args = p.parse_args()

    if args.train_all:
        seeds = args.seed if len(args.seed) > 1 or args.seed != [0] else ALL_SEEDS
        lrs = args.lr if len(args.lr) > 1 or args.lr != [0.1] else ALL_LRS
        wds = args.wd if len(args.wd) > 1 or args.wd != [1e-4] else ALL_WDS
        augs = args.aug if len(args.aug) > 1 or args.aug != ["basic"] else ALL_AUGS
        combos = list(itertools.product(seeds, lrs, wds, augs))
        print(f"Training {len(combos)} PlainCNN configurations...")
        for seed, lr, wd, aug in combos:
            run_training(seed, lr, wd, aug, args.epochs, args.gpu, args.save_dir, args.data_dir)
    else:
        for seed, lr, wd, aug in itertools.product(args.seed, args.lr, args.wd, args.aug):
            run_training(seed, lr, wd, aug, args.epochs, args.gpu, args.save_dir, args.data_dir)
