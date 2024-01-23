# -*- coding: utf-8 -*-
import argparse
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from utils import label_to_onehot, cross_entropy_for_onehot
from models.vision import LeNet, weights_init

import click


@click.command()
@click.option('--dataset', default="LFW", help='the dataset to use.')
@click.option('--img_index', default=25, help='the index for leaking images on CIFAR.')
@click.option('--image', default="", help='the path to customized image.')
@click.option('--img_size', default=64, help='the size of image.')
@click.option('--max_iter', default=3000, help='the max iteration for optimization.')
def main(dataset, img_index, image, img_size, max_iter):
    print(torch.__version__, torchvision.__version__)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print("Running on %s" % device)
    tp = transforms.ToTensor()
    tr = transforms.Resize((img_size, img_size))
    tt = transforms.ToPILImage()

    if len(image) > 1:
        gt_data = Image.open(image)
        gt_data = tp(gt_data).to(device)
    else:
        if dataset == "CIFAR100":
            dst = datasets.CIFAR100("~/.torch", download=True)
        elif dataset == "CIFAR10":
            dst = datasets.CIFAR10("~/.torch", download=True)
        elif dataset == "LFW":
            dst = datasets.LFWPeople("~/.torch", download=True)
        else:
            raise ValueError(f"{dataset} is Unknown dataset.")
        gt_data = tr(tp(dst[img_index][0])).to(device)

        print(gt_data.shape, dst[img_index][1])

    gt_data = gt_data.view(1, *gt_data.size())
    gt_label = torch.Tensor([1]).long().to(device)
    gt_label = gt_label.view(1, )
    gt_onehot_label = label_to_onehot(gt_label, num_classes=10)

    plt.imshow(tt(gt_data[0].cpu()))
    plt.pause(0.1)
    net = LeNet(num_classes=10, img_size=img_size).to(device)
    net = net.double()

    torch.manual_seed(55)
    net.apply(weights_init)
    criterion = cross_entropy_for_onehot

    # compute original gradient
    pred = net(gt_data.double())
    y = criterion(pred, gt_onehot_label)
    dy_dx = torch.autograd.grad(y, net.parameters())

    original_dy_dx = list((_.detach().clone() for _ in dy_dx))

    # generate dummy data and label
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    # dummy_data = tr(tp(dst[img_index+1][0])).to(device)
    # dummy_data = dummy_data.view(1, *dummy_data.size())
    dummy_data = dummy_data.to(device).requires_grad_(True)
    dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

    plt.imshow(tt(dummy_data[0].cpu()))

    optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=.1)

    history = []
    for iters in range(max_iter):

        def closure():
            optimizer.zero_grad()
            dummy_pred = net(dummy_data.double())
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = criterion(dummy_pred, dummy_onehot_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()

            return grad_diff

        optimizer.step(closure)
        if iters % (max_iter//30) == 0:
            current_loss = closure()
            print(iters, "%.6f" % current_loss.item())
            history.append(tt(dummy_data[0].cpu()))
            plt.imshow(history[-1])
            plt.pause(0.1)
            # if current_loss.item() < 1e-5:
            #     break

    plt.figure(figsize=(12, 8))
    for i in range(30):
        if i == len(history):
            break
        plt.subplot(3, 10, i + 1)
        plt.imshow(history[i])
        plt.title("iter=%d" % (i * 10))
        plt.axis('off')

    plt.show()


if __name__ == "__main__":
    main()
