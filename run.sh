#!/bin/sh
################ DAJAT 200 epochs RN18 CIFAR-10 #######################
CUDA_VISIBLE_DEVICES=0 python train_DAJAT.py --use_defaults CIFAR10_RN18  | tee -a train.txt
CUDA_VISIBLE_DEVICES=0 python eval.py --trained_model ./model-cifar-ResNet/ours-model-epoch200.pt  | tee -a eval_last_200.txt
CUDA_VISIBLE_DEVICES=0 python eval.py --trained_model ./model-cifar-ResNet/ours-model-epoch-SWA0.9996150200.pt  | tee -a eval_last_SWA_200.txt

################ DAJAT 200 epochs WRN-34-10 CIFAR-10 #######################
CUDA_VISIBLE_DEVICES=0 python train_DAJAT.py --use_defaults CIFAR10_WRN | tee -a train.txt
CUDA_VISIBLE_DEVICES=0 python eval.py --trained_model ./model-cifar-ResNet/ours-model-epoch200.pt  | tee -a eval_last_200.txt
CUDA_VISIBLE_DEVICES=0 python eval.py --trained_model ./model-cifar-ResNet/ours-model-epoch-SWA0.9996150200.pt  | tee -a eval_last_SWA_200.txt

################ DAJAT 200 epochs RN18 -CIFAR-100 #######################
CUDA_VISIBLE_DEVICES=0 python train_DAJAT.py  --use_defaults CIFAR100_RN18 | tee -a train.txt
CUDA_VISIBLE_DEVICES=0 python eval.py --trained_model ./model-cifar-ResNet/ours-model-epoch200.pt  | tee -a eval_last_200.txt
CUDA_VISIBLE_DEVICES=0 python eval.py --trained_model ./model-cifar-ResNet/ours-model-epoch-SWA0.9996150200.pt  | tee -a eval_last_SWA_200.txt

################ DAJAT 200 epochs WRN-34-10 -CIFAR-100 #######################
CUDA_VISIBLE_DEVICES=0 python train_DAJAT.py --use_defaults CIFAR100_WRN | tee -a train.txt
CUDA_VISIBLE_DEVICES=0 python eval.py --trained_model ./model-cifar-ResNet/ours-model-epoch200.pt  | tee -a eval_last_200.txt
CUDA_VISIBLE_DEVICES=0 python eval.py --trained_model ./model-cifar-ResNet/ours-model-epoch-SWA0.9996150200.pt  | tee -a eval_last_SWA_200.txt


################ ACAT 110 epochs RN18 CIFAR-10 #######################
CUDA_VISIBLE_DEVICES=0 python train_DAJAT.py --use_defaults CIFAR10_RN18 --epochs 110 --num_auto 0 | tee -a train.txt
CUDA_VISIBLE_DEVICES=0 python eval.py --trained_model ./model-cifar-ResNet/ours-model-epoch110.pt  | tee -a eval_last_110.txt
CUDA_VISIBLE_DEVICES=0 python eval.py --trained_model ./model-cifar-ResNet/ours-model-epoch-SWA0.999683110.pt  | tee -a eval_last_SWA_110.txt


################ ACAT 110 epochs WRN-34-10 CIFAR-10 #######################
CUDA_VISIBLE_DEVICES=0 python train_DAJAT.py --use_defaults CIFAR10_RN18 --epochs 110 --num_auto 0 | tee -a train.txt
CUDA_VISIBLE_DEVICES=0 python eval.py --trained_model ./model-cifar-ResNet/ours-model-epoch110.pt  | tee -a eval_last_110.txt
CUDA_VISIBLE_DEVICES=0 python eval.py --trained_model ./model-cifar-ResNet/ours-model-epoch-SWA0.999683110.pt  | tee -a eval_last_SWA_110.txt


################ ACAT 110 epochs RN18 CIFAR-100 #######################
CUDA_VISIBLE_DEVICES=0 python train_DAJAT.py --use_defaults CIFAR100_RN18 --epochs 110 --num_auto 0 | tee -a train.txt
CUDA_VISIBLE_DEVICES=0 python eval.py --trained_model ./model-cifar-ResNet/ours-model-epoch110.pt  | tee -a eval_last_110.txt
CUDA_VISIBLE_DEVICES=0 python eval.py --trained_model ./model-cifar-ResNet/ours-model-epoch-SWA0.999683110.pt  | tee -a eval_last_SWA_110.txt


################ ACAT 110 epochs WRN-34-10 CIFAR-100 #######################
CUDA_VISIBLE_DEVICES=0 python train_DAJAT.py --use_defaults CIFAR100_RN18 --epochs 110 --num_auto 0 | tee -a train.txt
CUDA_VISIBLE_DEVICES=0 python eval.py --trained_model ./model-cifar-ResNet/ours-model-epoch110.pt  | tee -a eval_last_110.txt
CUDA_VISIBLE_DEVICES=0 python eval.py --trained_model ./model-cifar-ResNet/ours-model-epoch-SWA0.999683110.pt  | tee -a eval_last_SWA_110.txt