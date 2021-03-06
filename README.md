# Efficient and Effective Augmentation Strategy for Adversarial Training
This repository contains codes for the training and evaluation of our CVPR 2022 Workshop paper [Efficient and Effective Strategy for Adversarial Training](https://artofrobust.github.io/short_paper/31.pdf).

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
THe GAMA-PGD-100 evaluation code is provided in eval.py.
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
