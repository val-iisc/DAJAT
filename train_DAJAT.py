from __future__ import print_function
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from utils import Bar, Logger, AverageMeter, accuracy
from utils_awp import TradesAWP
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from PIL import Image
from autoaugment import CIFAR10Policy
from models.resnet import ResNet18
import models
import torchvision
from defaults import use_default

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--arch', type=str, default='ResNet18')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='retrain from which epoch')
parser.add_argument('--data', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100'])
parser.add_argument('--data-path', type=str, default='./data',
                    help='where is the dataset CIFAR-10')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.2, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'],
                    help='The threat model')
parser.add_argument('--train_budget', default='high', type=str, choices=['low', 'high'],
                    help='The compute budget for training. High budget would mean larger number of atatck iterations')
parser.add_argument('--epsilon', default=8, type=float,
                    help='perturbation')
parser.add_argument('--step-size', default=8, type=float,
                    help='perturb step size')
parser.add_argument('--beta', default=11.0, type=float,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--LS', type=int, default=0, metavar='S',
                    help='make 1 is want to use Label Smoothing. DAJAT uses LS only for CIFAR100 dataset')
parser.add_argument('--model-dir', default='./model-cifar-ResNet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--resume-model', default='', type=str,
                    help='directory of model for retraining')
parser.add_argument('--resume-optim', default='', type=str,
                    help='directory of optimizer for retraining')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--num_auto', default=2, type=int, metavar='N',
                    help='Number of autoaugments to use for training')
parser.add_argument('--JS_weight', default=2, type=int, metavar='N',
                    help='The weight of the JS divergence term')
parser.add_argument('--awp-gamma', default=0.005, type=float,
                    help='whether or not to add parametric noise')
parser.add_argument('--awp-warmup', default=10, type=int,
                    help='We could apply AWP after some epochs for accelerating.')
parser.add_argument('--use_defaults', type=str, default='NONE' ,choices=['NONE','CIFAR10_RN18', 'CIFAR10_WRN','CIFAR100_WRN', 'CIFAR100_RN18'],
                    help='Use None is want to use the hyperparamters passed in the python training command else use the desired set of default hyperparameters')


args = parser.parse_args()
if args.use_defaults!='NONE':
    args = use_default(args.use_defaults)
print(args)

epsilon = args.epsilon / 255
args.epsilon = epsilon
if args.awp_gamma <= 0.0:
    args.awp_warmup = np.infty
if args.data == 'CIFAR100':
    NUM_CLASSES = 100
else:
    NUM_CLASSES = 10

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

class Get_Dataset_C10(torchvision.datasets.CIFAR10):

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        
        img, target = self.data[idx], self.targets[idx]
        image = Image.fromarray(img)
        image_clean = self.transform[0](image)
        image_auto1 = self.transform[1](image)
        image_auto2 = self.transform[1](image)
        return image_clean, image_auto1, image_auto2, target

class Get_Dataset_C100(torchvision.datasets.CIFAR100):

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        
        img, target = self.data[idx], self.targets[idx]
        image = Image.fromarray(img)
        image_clean = self.transform[0](image)
        image_auto1 = self.transform[1](image)
        image_auto2 = self.transform[1](image)
        return image_clean, image_auto1, image_auto2, target


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_auto = transforms.Compose([CIFAR10Policy(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

if args.data=='CIFAR10':
    trainset = Get_Dataset_C10(root=args.data_path, train=True,transform=[transform_train, transform_auto], download=True)
elif args.data=='CIFAR100':
    trainset = Get_Dataset_C100(root=args.data_path, train=True,transform=[transform_train, transform_auto], download=True)

testset = getattr(datasets, args.data)(root=args.data_path, train=False, download=True, transform=transform_test)
valset = getattr(datasets, args.data)(root=args.data_path, train=True, download=True, transform=transform_test)

train_size = 49000
valid_size = 1000
test_size  = 10000
train_indices = list(range(50000))
val_indices = []
count = np.zeros(100)
for index in range(len(trainset)):
    _,_,_, target = trainset[index]
    if(np.all(count==10)):
        break
    if(count[target]<10):
        count[target] += 1
        val_indices.append(index)
        train_indices.remove(index)


print("Overlap indices:",list(set(train_indices) & set(val_indices)))
print("Size of train set:",len(train_indices))
print("Size of val set:",len(val_indices))

train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,sampler=SubsetRandomSampler(train_indices), **kwargs)
val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,sampler=SubsetRandomSampler(val_indices), **kwargs)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=100, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))



def perturb_input(model,
                  x_natural,
                  step_size=0.003,
                  epsilon=0.031,
                  perturb_steps=10,
                  distance='l_inf', batch_norm='base'):
    model.eval()
    batch_size = len(x_natural)
    if distance == 'l_inf':
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = F.kl_div(F.log_softmax(model(x_adv, batch_norm), dim=1),
                                   F.softmax(model(x_natural, batch_norm), dim=1),
                                   reduction='sum')
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv


def train(model, train_loader, optimizer, epoch, awp_adversary, start_wa, tau_list, exp_avgs):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    print('epoch: {}'.format(epoch))
    bar = Bar('Processing', max=len(train_loader))

    for batch_idx, (data_base, data_auto1, data_auto2, target) in enumerate(train_loader):
        x_base, x_auto1, x_auto2, target = data_base.to(device), data_auto1.cuda(), data_auto2.cuda(), target.to(device)
        varepsilon = args.epsilon*(epoch/args.epochs)
        if args.train_budget=='low':
            step_size = varepsilon
            iters_attack = 2
        elif args.train_budget=='high':
            if epoch<=50:
                step_size = varepsilon
                iters_attack = 2
            if epoch<=100:
                step_size = 2*varepsilon/3
                iters_attack = 3
            if epoch<=150:
                step_size = varepsilon/2
                iters_attack = 4
            if epoch<=200:
                step_size = varepsilon/2
                iters_attack = 5

        x_adv_base = perturb_input(model=model,
                              x_natural=x_base,
                              step_size=step_size,
                              epsilon=varepsilon,
                              perturb_steps=iters_attack,
                              distance=args.norm, batch_norm='base')
        if args.num_auto>=1:
            x_adv_auto1 = perturb_input(model=model,
                                  x_natural=x_auto1,
                                  step_size=step_size,
                                  epsilon=varepsilon,
                                  perturb_steps=iters_attack,
                                  distance=args.norm, batch_norm='auto')
        if args.num_auto>=2:
            x_adv_auto2 = perturb_input(model=model,
                                  x_natural=x_auto2,
                                  step_size=step_size,
                                  epsilon=varepsilon,
                                  perturb_steps=iters_attack,
                                  distance=args.norm, batch_norm='auto')

        model.train()
        if epoch >= args.awp_warmup:
            awp = awp_adversary.calc_awp(inputs_adv=x_adv_base,
                                         inputs_clean=x_base,
                                         targets=target,
                                         beta=args.beta)
            awp_adversary.perturb(awp)

        optimizer.zero_grad()
        logits_base = model(x_base)
        logits_adv_base = model(x_adv_base)
        if args.LS==1:
            criterion = LabelSmoothingLoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()
        loss_robust_base = F.kl_div(F.log_softmax(logits_adv_base, dim=1),
                               F.softmax(logits_base, dim=1),
                               reduction='batchmean')
        p_base = F.softmax(logits_base, dim=1)
        loss_natural_base = criterion(logits_base, target)
        loss_base = loss_natural_base + args.beta * loss_robust_base
        if args.num_auto==0:
            loss = loss_base

        if args.num_auto>=1:
            logits_auto1 = model(x_auto1, 'auto')
            logits_adv_auto1 = model(x_adv_auto1, 'auto')
            loss_robust_auto1 = F.kl_div(F.log_softmax(logits_adv_auto1, dim=1),
                                   F.softmax(logits_auto1, dim=1),
                                   reduction='batchmean')
            p_auto1 = F.softmax(logits_auto1, dim=1)
            loss_natural_auto1 = criterion(logits_auto1, target)
            loss_auto1 = loss_natural_auto1 + args.beta * loss_robust_auto1
            if args.num_auto==1:
                p_mixture = torch.clamp((p_base + p_auto1) / 2., 1e-7, 1).log()
                loss_JS = (F.kl_div(p_mixture, p_base, reduction='batchmean') + F.kl_div(p_mixture, p_auto1, reduction='batchmean') )/2
                loss = (loss_base + loss_auto1)/2 + args.JS_weight*loss_JS

        if args.num_auto>=2:
            logits_auto2 = model(x_auto2, 'auto')
            logits_adv_auto2 = model(x_adv_auto2, 'auto')
            loss_robust_auto2 = F.kl_div(F.log_softmax(logits_adv_auto2, dim=1),
                                   F.softmax(logits_auto2, dim=1),
                                   reduction='batchmean')
            p_auto2 = F.softmax(logits_auto2, dim=1)
            loss_natural_auto2 = criterion(logits_auto2, target)
            loss_auto2 = loss_natural_auto2 + args.beta * loss_robust_auto2

            p_mixture = torch.clamp((p_base + p_auto1 + p_auto2) / 3., 1e-7, 1).log()
            loss_JS = (F.kl_div(p_mixture, p_base, reduction='batchmean') + F.kl_div(p_mixture, p_auto1, reduction='batchmean') + F.kl_div(p_mixture, p_auto2, reduction='batchmean') )/3
            loss = (loss_base + loss_auto1 + loss_auto2)/3 + args.JS_weight*loss_JS


        prec1, prec5 = accuracy(logits_adv_base, target, topk=(1, 5))
        losses.update(loss.item(), x_base.size(0))
        top1.update(prec1.item(), x_base.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= args.awp_warmup:
            awp_adversary.restore(awp)

        batch_time.update(time.time() - end)
        end = time.time()

        for start_ep, tau, new_state_dict in zip(start_wa, tau_list, exp_avgs):
            if epoch == start_ep:
                for key,value in model.state_dict().items():
                    new_state_dict[key] = value
            elif epoch > start_ep:
                for key,value in model.state_dict().items():
                    new_state_dict[key] = (1-tau)*value + tau*new_state_dict[key]
            else:
                pass


        bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s| Total:{total:}| ETA:{eta:}| Loss:{loss:.4f}| top1:{top1:.2f}'.format(
            batch=batch_idx + 1,
            size=len(train_loader),
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg)
        bar.next()
    bar.finish()
    return losses.avg, top1.avg, exp_avgs


def test(model, test_loader, criterion):
    global best_acc
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(test_loader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s| Total: {total:}| ETA: {eta:}| Loss:{loss:.4f}| top1: {top1:.2f}'.format(
                batch=batch_idx + 1,
                size=len(test_loader),
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg)
            bar.next()
    bar.finish()
    return losses.avg, top1.avg



def adjust_learning_rate_cosine(optimizer, epoch, args):
    lr = args.lr * 0.5 * (1 + np.cos((epoch - 1) / args.epochs * np.pi))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def main():
    ################### Change here to WideResNet34 if you want to train on WRN-34-10 #############################
    model = nn.DataParallel(getattr(models, args.arch)(num_classes=NUM_CLASSES)).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    ################### Change here to WideResNet34 if you want to train on WRN-34-10 #############################
    proxy = nn.DataParallel(getattr(models, args.arch)(num_classes=NUM_CLASSES)).to(device)
    proxy_optim = optim.SGD(proxy.parameters(), lr=args.lr)
    awp_adversary = TradesAWP(model=model, proxy=proxy, proxy_optim=proxy_optim, gamma=args.awp_gamma)

    criterion = nn.CrossEntropyLoss()

    logger = Logger(os.path.join(model_dir, 'log.txt'), title=args.arch)
    logger.set_names(['Learning Rate', 'Nat Val Loss', 'Nat Val Acc.'])

    if args.resume_model:
        model.load_state_dict(torch.load(args.resume_model, map_location=device))
    if args.resume_optim:
        optimizer.load_state_dict(torch.load(args.resume_optim, map_location=device))
    start_wa = [(150*args.epochs)//200]
    tau_list = [0.9996]
    exp_avgs = []
    model_tau = nn.DataParallel(getattr(models, args.arch)(num_classes=NUM_CLASSES)).to(device)
    exp_avgs.append(model_tau.state_dict())
    
    for epoch in range(args.start_epoch, args.epochs + 1):
        lr = adjust_learning_rate_cosine(optimizer, epoch, args)


        adv_loss, adv_acc, exp_avgs = train(model, train_loader, optimizer, epoch, awp_adversary, start_wa, tau_list, exp_avgs)

        print('================================================================')
        val_loss, val_acc = test(model, test_loader, criterion)
        print('================================================================')

        logger.append([lr, val_loss, val_acc])

        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'ours-model-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'ours-opt-checkpoint_epoch{}.tar'.format(epoch)))

        if epoch >=args.epochs-1:
            for idx, start_ep, tau, new_state_dict in zip(range(len(tau_list)), start_wa, tau_list, exp_avgs):
                if start_ep <= epoch:
                    torch.save(new_state_dict,os.path.join(model_dir, 'ours-model-epoch-SWA{}{}{}.pt'.format(tau,start_ep,epoch)))
            

if __name__ == '__main__':
    main()
