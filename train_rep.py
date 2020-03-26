import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
# from rep_model import SSD300RepPoint, RepPointLoss
from RepPointsModel import SSD300RepPoint, RepPointLoss
from datasets import PascalVOCDataset
from utils import *
from tqdm import tqdm
from pprint import PrettyPrinter
import numpy as np

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Data parameters
data_folder = 'data'  # folder with data files
f_log = open('log/rep_train_log.txt', 'w')
keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
checkpoint = None  # path to model checkpoint, None if none
batch_size = 32  # batch size
internal_batchsize = 2
num_iter_flag = batch_size // internal_batchsize

iterations = 100000 * num_iter_flag  # number of iterations to train
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 3200  # print training status every __ batches
lr = 1e-3  # learning rate
decay_lr_at = [70000 * num_iter_flag, 90000 * num_iter_flag]  # decay learning rate after these many iterations
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 1e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

cudnn.benchmark = True


def main():
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at, best_mAP

    # Initialize model or load checkpoint
    if checkpoint is None:
        start_epoch = 0
        model = SSD300RepPoint(n_classes=n_classes, center_init=False, transform_method='min-max')
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    criterion = RepPointLoss(rep_point_xy=model.rep_points_xy, scale_weights=model.weights).to(device)

    # Custom dataloaders
    train_dataset = PascalVOCDataset(data_folder,
                                     split='train',
                                     keep_difficult=keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=internal_batchsize, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here

    # Load test data
    test_dataset = PascalVOCDataset(data_folder,
                                    split='test',
                                    keep_difficult=keep_difficult)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=internal_batchsize, shuffle=False,
                                              collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)

    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations

    epochs = iterations // (len(train_dataset) // internal_batchsize)
    # print('Length of dataset:', len(train_dataset), epochs)
    decay_lr_at = [it // (len(train_dataset) // internal_batchsize) for it in decay_lr_at]
    print('total train epochs: ', epochs)

    # Epochs
    best_mAP = -1.
    # criterion.increase_threshold()
    # print('current threshold: ', criterion.threshold)
    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        # Save checkpoint
        if epoch >= 80 and epoch % 30 == 0:
            _, current_mAP = evaluate(test_loader, model)
            if current_mAP > best_mAP:
                save_checkpoint(epoch, model, optimizer, name='checkpoints/my_checkpoint_rep300_b32.pth.tar')
                best_mAP = current_mAP
                # criterion.increase_threshold(0.05)
        elif epoch == 50:
            save_checkpoint(epoch, model, optimizer, name='checkpoints/my_checkpoint_rep300_b32.pth.tar')

    _, current_mAP = evaluate(test_loader, model)
    if current_mAP > best_mAP:
        save_checkpoint(epoch, model, optimizer, name='checkpoints/my_checkpoint_rep300_b32.pth.tar')
        best_mAP = current_mAP


def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()
    optimizer.zero_grad()

    # Batches
    # loss_mini_batch = list()
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs_init, predicted_locs_refine, predicted_init, predicted_scores, _ = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs_init, predicted_locs_refine, predicted_init,
                         predicted_scores, boxes, labels) / num_iter_flag  # scalar

        # Backward prop.
        if i % num_iter_flag == 0 and i != 0:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        else:
            loss.backward()

        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()


        # # Clip gradients, if necessary
        # if grad_clip is not None:
        #     clip_gradient(optimizer, grad_clip)

        losses.update(loss.item() * num_iter_flag, images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            str_print = 'Epoch: [{0}][{1}/{2}]\t' \
                        'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\n'.format(epoch, i, len(train_loader),
                                                                        batch_time=batch_time,
                                                                        data_time=data_time, loss=losses)
            print(str_print.strip('\n'))
            f_log.write(str_print)
            # print('Epoch: [{0}][{1}/{2}]\t'
            #       'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #       'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
            #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
            #                                                       batch_time=batch_time,
            #                                                       data_time=data_time, loss=losses))
    del predicted_locs_init, predicted_locs_refine, predicted_init, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored


def evaluate(test_loader, model):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py
    detect_speed = list()

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)
            # print('batch size in current iter: ', len(labels))

            # Forward prop.
            time_start = time.time()
            _, predicted_locs, _, predicted_scores, _ = model(images)
            time_end = time.time()


            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.01, max_overlap=0.5,
                                                                                       top_k=100)

            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)
            detect_speed.append((time_end - time_start) / len(labels))

        # Calculate mAP
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)

    # Print AP for each class
    pp.pprint(APs)

    # added to resume training
    model.train()

    str_print = 'EVAL: Mean Average Precision {0:.3f}, avg speed {1:.2f} Hz\n'.format(mAP, 1. / np.mean(detect_speed))
    print(str_print)
    f_log.write(str_print)
    # print('Mean Average Precision (mAP): %.3f\n' % mAP)

    return APs, mAP


if __name__ == '__main__':
    main()
