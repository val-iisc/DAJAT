# Efficient and Effective Augmentation Strategy for Adversarial Training
This repository contains codes for the training and evaluation of our NeurIPS-22 paper  [Efficient and Effective Strategy for Adversarial Training](https://arxiv.org/abs/2210.15318). The openreview link for the paper is also  [available](https://openreview.net/forum?id=ODkBI1d3phW).
![plot](./DAJAT_fig.png)
# Training
For training DAJAT: 
```
python train_DAJAT.py --use_defaults ['NONE','CIFAR10_RN18', 'CIFAR10_WRN','CIFAR100_WRN', 'CIFAR100_RN18']
```
For training ACAT: 
```
python train_DAJAT.py --use_defaults ['NONE','CIFAR10_RN18', 'CIFAR10_WRN','CIFAR100_WRN', 'CIFAR100_RN18']  --num_autos 0 --epochs 110 --beta
```
# Evaluation
The GAMA-PGD-100 evaluation code is provided in eval.py.
For evaluation of the trained model: 
```
python eval.py --trained_model 'PATH OF TRAINED MODEL' 
```
Further all the running details are provided in run.sh. It is recommended to use this file for training and evaluation of DAJAT.

# Results
![plot](./DAJAT_C10.png)
![plot](./DAJAT_C100.png)

Results obtained using higher number of attack steps and 200 epochs for training:
![plot](./DAJAT_200.png)
