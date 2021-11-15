import sys
sys.path.append('..')
import os
import torch
import torch.nn.parallel
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms as transforms
import random
import copy
import argparse
from tqdm import tqdm
import numpy as np
from termcolor import cprint
from utils import lrt_correction
from networks.preact_resnet import preact_resnet34
from noise import noisify_with_P, noisify_cifar10_asymmetric, noisify_cifar100_asymmetric
from cifar_dataset import CIFAR10, CIFAR100

def _init_fn(worker_id):
    np.random.seed(77 + worker_id)

def main(args):
    # Initialize random seeds
    random_seed = args.seed
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True

    # Load the noisy label file
    noise_label_path = os.path.join('labels', args.noise_label_file)
    noisy_labels = np.load(noise_label_path)

    # Select the decive (CPU or CUDA)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set hyperparameters
    batch_size: int = 128
    num_workers: int = 2
    train_val_ratio: float = 0.9
    lr: float = 0.01
    current_thd: float = 0.3
    thd_increment: float = 0.1
        
    dataset_name = args.noise_label_file.split('-')[0]
    
    # Data augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    transform_test = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])


    if dataset_name == 'cifar10':
        trainset = CIFAR10(root = 'data', split = 'train', train_ratio = train_val_ratio, download = True, transform = transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = True, num_workers = num_workers, worker_init_fn = _init_fn)

        valset = CIFAR10(root = 'data', split = 'val', train_ratio = train_val_ratio, download = True, transform = transform_test)
        valloader = torch.utils.data.DataLoader(valset, batch_size = batch_size, shuffle = False, num_workers = num_workers)

        testset = CIFAR10(root = 'data', split = 'test', download = True, transform = transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = num_workers)

        num_class = 10

    elif dataset_name == 'cifar100':
        trainset = CIFAR100(root = 'data', split = 'train', train_ratio = train_val_ratio, download = True, transform = transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = True, num_workers = num_workers, worker_init_fn = _init_fn)

        valset = CIFAR100(root = 'data', split = 'val', train_ratio = train_val_ratio, download = True, transform = transform_test)
        valloader = torch.utils.data.DataLoader(valset, batch_size = batch_size, shuffle = False, num_workers = num_workers)

        testset = CIFAR100(root = 'data', split = 'test', download = True, transform = transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = num_workers)

        num_class = 100

    else:
        raise ValueError('Dataset should be CIFAR10 or CIFAR100.')

    print('\nTrain set size:', len(trainset))
    print('Valid set size:', len(valset))
    print('Test set size:', len(testset), '\n')

    # Generate noise and add it to labels
    y_clean = copy.deepcopy(trainset.get_data_labels())
    y_train = noisy_labels.copy()
    p = None

    if args.noise_type == "uniform" and args.noise_rate > 0.0:
        y_train, p = noisify_with_P(y_train, nb_classes = num_class, noise = args.noise_rate, random_state = random_seed)
        print("applied uniform noise")
        print("probability transition matrix:\n{}\n".format(p))

    elif args.noise_type == "asym" and args.noise_rate > 0.0:
        if dataset_name == 'cifar10':
            y_train, p = noisify_cifar10_asymmetric(y_train, noise = args.noise_rate, random_state = random_seed)
        elif dataset_name == 'cifar100':
            y_train, p = noisify_cifar100_asymmetric(y_train, noise = args.noise_rate, random_state = random_seed)
        print("applied asymmetric noise")
        print("probability transition matrix:\n{}\n".format(p))

    trainset.update_corrupted_label(y_train)

    real_noise_rate = round(np.sum(y_train != y_clean) / len(y_train),3)
    print('>> Real Noise Level: {:.3f}\n'.format(real_noise_rate))

    # Setup network
    model = preact_resnet34(num_classes = num_class)

    print("============= Start Training =============")

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.9, nesterov = True, weight_decay = 5e-4)
    scheduler = MultiStepLR(optimizer, milestones = [40, 80], gamma = 0.5)
    model = model.to(device)

    output_record = torch.zeros([30, len(y_train), num_class])
    output_selected = torch.zeros([5, len(y_train), num_class])
    val_record = torch.zeros([30])

    best_acc = 0
    best_epoch = 0
    best_weights = None

    for epoch in range(args.nepoch):
        # Training
        train_loss = 0
        train_correct = 0
        train_total = 0

        model.train()
        for _, (features, labels, indices) in enumerate(tqdm(trainloader, ascii = True, ncols = 50)):
            if features.shape[0] == 1:
                continue

            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_total += features.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            output_record[epoch % 30, indices] = F.softmax(outputs.detach().cpu(), dim = 1)
        np.save('output_{}.csv'.format(epoch), output_record[epoch % 30])

        train_acc = train_correct / train_total * 100

        # Validation
        val_total = 0
        val_correct = 0
        model.eval()
        with torch.no_grad():
            for i, (images, labels, _) in enumerate(valloader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_total += images.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = val_correct / val_total * 100.
        val_record[epoch % 30] = val_acc

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())

        ind = np.argsort(val_record)[-5:]

        cprint("Epoch {}|{}. Train accuracy: {:.2f}%  Val accuracy: {:.2f}".format(epoch + 1, args.nepoch, train_acc, val_acc), "yellow")
        cprint('>> Top 5 accuracies: {}'.format(np.array(val_record)[ind]), 'green')
        
        # Correction
        if epoch >= 15:
            output_selected = output_record[ind].mean(0)
            y_corrected, current_thd = lrt_correction(np.array(trainset.targets).copy(), output_selected, current_thd = current_thd, thd_increment = thd_increment)
            trainset.update_corrupted_label(y_corrected)

        scheduler.step()

    # Testing
    test_total = 0
    test_correct = 0
    model.load_state_dict(best_weights)
    model.eval()

    with torch.no_grad():
        for i, (images, labels, _) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_total += images.size(0)
            _, predicted = outputs.max(1)
            test_correct += predicted.eq(labels).sum().item()

    test_acc = test_correct / test_total * 100.
    cprint('>> Best test accuracy: {:.2f}%'.format(test_acc), 'cyan')
    cprint('>> Final noise rate {:.3f}%'.format(100 - sum(np.array(trainset.targets).flatten() == np.array(y_clean).flatten()) / float(len(np.array(y_train))) * 100), 'cyan')
    return test_acc

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_label_file", default = 'cifar10-1-0.35.npy', help = 'noise label file', type = str)
    parser.add_argument("--nepoch", default = 180, help = 'number of training epochs', type = int)
    parser.add_argument("--noise_type", default = 'uniform', help = 'noise type (uniform or asym)', type = str)
    parser.add_argument("--noise_rate", default = 0.0, help = 'noise rate for CCN', type = float
    parser.add_argument("--seed", default = 77, help = 'random seed', type = int)
    args = parser.parse_args()
    main(args)                        
