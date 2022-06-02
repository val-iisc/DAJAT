import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import time
import math
import argparse
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import torch.nn as nn
import torch.optim as optim
import models
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from models.resnet import ResNet18
import models 

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--trained_model', default='./',
                    help='location of the adversarially trained model')
parser.add_argument('--arch', type=str, default='ResNet18')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--data', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100'])
parser.add_argument('--data-path', type=str, default='./data',
                    help='where is the dataset')
parser.add_argument('--epsilon1', default=8/255, type=float,
                    help='perturbation')
parser.add_argument('--epsilon2', default=12/255, type=float,
                    help='perturbation')
parser.add_argument('--epsilon3', default=16/255, type=float,
                    help='perturbation')
parser.add_argument('--use_GAMA_epsilon1', action='store_true', default=True,
                    help='perturbation')
parser.add_argument('--use_GAMA_epsilon2', action='store_true', default=False,
                    help='perturbation')
parser.add_argument('--use_GAMA_epsilon3',action='store_true', default=False,
                    help='perturbation')
parser.add_argument('--use_BB_attack',action='store_true', default=False,
                    help='perturbation')
parser.add_argument('--model_std', type=str, default='./',
                    help='where is the standard trained model')
parser.add_argument('--run_rfgsm',action='store_true', default=False,
                    help='perturbation')
parser.add_argument('--run_bbfgsm',action='store_true', default=False,
                    help='perturbation')
args = parser.parse_args()


#loading data 
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

if args.data == 'CIFAR10' or args.data == 'CIFAR100':
    testset = getattr(datasets, args.data)(root=args.data_path, train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

if args.data == 'CIFAR10':
    NUM_CLASSES = 10
    test_size = 10000
elif args.data == 'CIFAR100':
    NUM_CLASSES = 100
    test_size = 10000

##################################### Load std trained model #############################
model_std = nn.DataParallel(getattr(models, args.arch)(num_classes=NUM_CLASSES)).cuda()
model_std.cuda()
if args.use_BB_attack:
    model_std = nn.DataParallel(model_std)
    model_dict = torch.load(args.model_std)
    model_std.load_state_dict(model_dict)            

##################################### Load adv trained model #############################
model = nn.DataParallel(getattr(models, args.arch)(num_classes=NUM_CLASSES)).cuda()
model_dict = torch.load(args.trained_model)  
model.load_state_dict(model_dict)

model_std.eval()
model.eval()



def normalize(X):
    return (X)
    
def R_FGSM_Attack_step(model,loss,image,target,eps,bounds,steps=1):
    #Raise error if in training mode
    assert not model.training, 'Model is in  training mode'
    tar = Variable(target.cuda())
    img = image.cuda()
    eps = eps/steps
    true_img = img
    B,C,H,W = img.size()
    noise = torch.FloatTensor(np.random.uniform(-eps,eps,(B,C,H,W))).cuda()
    img = torch.clamp(img+noise,bounds[0],bounds[1])
    for step in range(steps):
        img = Variable(img,requires_grad=True)
        zero_gradients(img) 
        out  = model((img))
        cost = loss(out,tar)
        cost.backward()
        per = torch.clamp(noise + eps * torch.sign(img.grad.data),-eps,eps)
        adv = true_img.data + per.cuda() 
        img = torch.clamp(adv,bounds[0],bounds[1])
    return img


    
def BB_FGSM_Attack_step(model,loss,image,target,eps,bounds,steps=1):
    #Raise error if in training mode
    assert not model.training, 'Model is in  training mode'
    tar = Variable(target.cuda())
    img = image.cuda()
    eps = eps/steps
    for step in range(steps):
        img = Variable(img,requires_grad=True)
        zero_gradients(img) 
        out  = model((img))
        cost = loss(out,tar)
        cost.backward()
        per = eps * torch.sign(img.grad.data)
        adv = img.data + per.cuda() 
        img = torch.clamp(adv,bounds[0],bounds[1])
    return img
    
    
def max_margin_loss(x,y):
    B = y.size(0)
    corr = x[range(B),y]

    x_new = x - 1000*torch.eye(NUM_CLASSES)[y].cuda()
    tar = x[range(B),x_new.argmax(dim=1)]
    loss = tar - corr
    loss = torch.mean(loss)
    
    return loss



def GAMA_PGD(model,data,target,eps,eps_iter,bounds,steps,w_reg,lin,SCHED,drop):
    """
    model
    loss : loss used for training
    data : input to network
    target : ground truth label corresponding to data
    eps : perturbation srength added to image
    eps_iter
    """
    #Raise error if in training mode
    if model.training:
        assert 'Model is in  training mode'
    tar = Variable(target.cuda())
    data = data.cuda()
    B,C,H,W = data.size()
    noise  = torch.FloatTensor(np.random.uniform(-eps,eps,(B,C,H,W))).cuda()
    noise  = eps*torch.sign(noise)
    img_arr = []
    W_REG = w_reg
    orig_img = data+noise
    orig_img = Variable(orig_img,requires_grad=True)
    for step in range(steps):
        # convert data and corresponding into cuda variable
        img = data + noise
        img = Variable(img,requires_grad=True)
        
        if step in SCHED:
            eps_iter /= drop
        
        # make gradient of img to zeros
        zero_gradients(img) 
        # forward pass        
        orig_out = model((orig_img))
        P_out = nn.Softmax(dim=1)(orig_out)
        
        out  = model((img))
        Q_out = nn.Softmax(dim=1)(out)
        #compute loss using true label
        if step <= lin:
            cost =  W_REG*((P_out - Q_out)**2.0).sum(1).mean(0) + max_margin_loss(Q_out,tar)
            W_REG -= w_reg/lin
        else:
            cost = max_margin_loss(Q_out,tar)
        #backward pass
        cost.backward()
        #get gradient of loss wrt data
        per =  torch.sign(img.grad.data)
        #convert eps 0-1 range to per channel range 
        per[:,0,:,:] = (eps_iter * (bounds[0,1] - bounds[0,0])) * per[:,0,:,:]
        if(per.size(1)>1):
            per[:,1,:,:] = (eps_iter * (bounds[1,1] - bounds[1,0])) * per[:,1,:,:]
            per[:,2,:,:] = (eps_iter * (bounds[2,1] - bounds[2,0])) * per[:,2,:,:]
        #  ascent
        adv = img.data + per.cuda()
        #clip per channel data out of the range
        img.requires_grad =False
        img[:,0,:,:] = torch.clamp(adv[:,0,:,:],bounds[0,0],bounds[0,1])
        if(per.size(1)>1):
            img[:,1,:,:] = torch.clamp(adv[:,1,:,:],bounds[1,0],bounds[1,1])
            img[:,2,:,:] = torch.clamp(adv[:,2,:,:],bounds[2,0],bounds[2,1])
        img = img.data
        noise = img - data
        noise  = torch.clamp(noise,-eps,eps)

    return data + noise


acc = 0
for batch_idx, (inputs, targets) in enumerate(test_loader):
  with torch.no_grad():
    inputs = inputs.cuda()
    targets = targets.cuda()
    outputs1 = model((inputs))
    acc+=torch.sum(torch.argmax(outputs1,dim=1)==targets.cuda())

acc = acc.detach().cpu().numpy()

print("Clean Accuracy: ",100*(acc/test_size))


loss=nn.CrossEntropyLoss()

if args.run_rfgsm:
    print("############################################################## RUNNING RFGSM ATTACK #######################################################################")


    for eps in [16/255,32/255]:
        loss = nn.CrossEntropyLoss()
        acc = 0
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data = inputs.cuda()
            target = targets.cuda()
            adv_img = R_FGSM_Attack_step(model,loss,data,target,eps,[0,1])

            acc+=torch.sum(torch.argmax(model(adv_img),dim=1)==targets.cuda())
        acc = acc.detach().cpu().numpy()
        print("RFGSM eps {} accuracy is {}".format(eps,100*(acc/test_size)))



if args.run_bbfgsm:
    print("############################################################## RUNNING BB-FGSM ATTACK #######################################################################")


    for eps in [16/255,32/255]:
        loss = nn.CrossEntropyLoss()
        acc = 0
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data = inputs.cuda()
            target = targets.cuda()
            adv_img = BB_FGSM_Attack_step(model_std,loss,data,target,eps,[0,1])
            acc+=torch.sum(torch.argmax(model(adv_img),dim=1)==targets.cuda())
        acc = acc.detach().cpu().numpy()
        print("BB-FGSM eps {} accuracy is {}".format(eps,100*(acc/test_size)))




print("############################################################## RUNNING GAMA=PGD ATTACK #######################################################################")

lst_eps=[]
if args.use_GAMA_epsilon1 == True:
    lst_eps.append(args.epsilon1)
if args.use_GAMA_epsilon2 == True:
    lst_eps.append(args.epsilon2)
if args.use_GAMA_epsilon3 == True:
    lst_eps.append(args.epsilon3)

for eps in lst_eps:
    steps=100
    loss = nn.CrossEntropyLoss()
    acc=0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        data = inputs.cuda()
        target = targets.cuda()
    
        with torch.enable_grad(): 
            adv_img = GAMA_PGD(model,data.cuda(),target.cuda(),eps=eps,eps_iter=2*eps,bounds=np.array([[0,1],[0,1],[0,1]]),steps=steps,w_reg=50,lin=25,SCHED=[60,85],drop=10)

        acc+=torch.sum(torch.argmax(model((adv_img)),dim=1)==targets.cuda())

    acc = acc.detach().cpu().numpy()
    print("GAMA-PGD-100 eps {} accuracy is {}".format(eps,100*(acc/test_size)))
