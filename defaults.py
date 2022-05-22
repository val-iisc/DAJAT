import argparse
def use_default(default_arg):
    parser = argparse.ArgumentParser(description='PyTorch DAJAT Adversarial Training')
    if default_arg == "CIFAR10_RN18":
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



    elif default_arg == "CIFAR10_WRN":
        parser.add_argument('--arch', type=str, default='WideResNet34')
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
        parser.add_argument('--beta', default=10.0, type=float,
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
        parser.add_argument('--JS_weight', default=3, type=int, metavar='N',
                            help='The weight of the JS divergence term')
        parser.add_argument('--awp-gamma', default=0.005, type=float,
                            help='whether or not to add parametric noise')
        parser.add_argument('--awp-warmup', default=10, type=int,
                            help='We could apply AWP after some epochs for accelerating.')

    elif default_arg == "CIFAR100_WRN":
        parser.add_argument('--arch', type=str, default='WideResNet34')
        parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                            help='input batch size for training (default: 128)')
        parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                            help='input batch size for testing (default: 128)')
        parser.add_argument('--epochs', type=int, default=200, metavar='N',
                            help='number of epochs to train')
        parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                            help='retrain from which epoch')
        parser.add_argument('--data', type=str, default='CIFAR100', choices=['CIFAR10', 'CIFAR100'])
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
        parser.add_argument('--beta', default=9.0, type=float,
                            help='regularization, i.e., 1/lambda in TRADES')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        parser.add_argument('--LS', type=int, default=1, metavar='S',
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

    elif default_arg == "CIFAR100_RN18":

        parser.add_argument('--arch', type=str, default='ResNet18')
        parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                            help='input batch size for training (default: 128)')
        parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                            help='input batch size for testing (default: 128)')
        parser.add_argument('--epochs', type=int, default=200, metavar='N',
                            help='number of epochs to train')
        parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                            help='retrain from which epoch')
        parser.add_argument('--data', type=str, default='CIFAR100', choices=['CIFAR10', 'CIFAR100'])
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
        parser.add_argument('--beta', default=9.0, type=float,
                            help='regularization, i.e., 1/lambda in TRADES')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        parser.add_argument('--LS', type=int, default=1, metavar='S',
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

    else:
        print("Use_default not Found")
        exit

    parser.add_argument('--use_defaults', type=str, default='CIFAR10_RN18' ,choices=['NONE','CIFAR10_RN18', 'CIFAR10_WRN','CIFAR100_WRN', 'CIFAR100_RN18'],help='Use None is want to use the hyperparamters passed in the python training command else use the desired set of default hyperparameters')
    args = parser.parse_args()
    return args