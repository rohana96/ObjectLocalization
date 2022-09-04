from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import sklearn
import torch
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import numpy as np
from datetime import datetime
import pickle as pkl

# imports
from task_1 import USE_WANDB
from wsddn import WSDDN
from voc_dataset import *
import wandb
from utils import nms, iou, tensor_to_PIL
from utils import *
from PIL import Image, ImageDraw
from AlexNet import LocalizerAlexNet
from task_1 import AverageMeter

# hyper-parameters
# ------------
start_step = 0
end_step = 20000
lr_decay_steps = {150000}
lr_decay = 1. / 10
rand_seed = 1024

lr = 0.0001
momentum = 0.9
weight_decay = 0.0005
# ------------

USE_WANDB = True
disp_interval = 10
val_interval = 500
train_dataset = VOCDataset('trainval', 512, top_n=300)
val_dataset = VOCDataset('test', 512, top_n=300)
train_step = 0
val_step = 0



if USE_WANDB:
    wandb.init(project="vlr2-hw2-task2", name="wsdnn_4096_0_4", reinit=True)
def main():
    if rand_seed is not None:
        np.random.seed(rand_seed)

    # load datasets and create dataloaders

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,  # batchsize is one for this implementation
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        sampler=None,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True)

    # Create network and initialize
    net = WSDDN(classes=train_dataset.CLASS_NAMES)
    print(net)

    if os.path.exists('pretrained_alexnet.pkl'):
        pret_net = pkl.load(open('pretrained_alexnet.pkl', 'rb'))
    else:
        pret_net = model_zoo.load_url(
            'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
        pkl.dump(pret_net, open('pretrained_alexnet.pkl', 'wb'),
                 pkl.HIGHEST_PROTOCOL)
    own_state = net.state_dict()

    for name, param in pret_net.items():
        print(name)
        if name not in own_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        try:
            own_state[name].copy_(param)
            print('Copied {}'.format(name))
        except:
            print('Did not find {}'.format(name))
            continue

    # Move model to GPU and set train mode
    net.load_state_dict(own_state)
    net.cuda()
    net.train()

    # TODO: Create optimizer for network parameters from conv2 onwards
    # for name, param in net.named_parameters():
    #     if 'features.0' in name:
    #         param.requires_grad = False
    for param in net.features.parameters():
        param.requires_grad = False
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    # (do not optimize conv1)

    output_dir = "./"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train(net, train_loader, val_loader, optimizer)


def visualize_bbox(image, classes, bboxes, scores, epoch, iter, eval="Train"):
    scores = list(np.array(scores, dtype = "float64"))
    class_id_to_label = dict(enumerate(val_dataset.CLASS_NAMES))
    boxes = get_box_data(classes, bboxes, scores)
    img = wandb.Image(image, boxes={
        "predictions": {
            "box_data": boxes,
            "class_labels": class_id_to_label,
        }
    })
    wandb.log({ 
        f'{eval}/Image': img,
        f'{eval}/Epoch': epoch
        })


def compute_ap(
        pred_boxes,
        pred_scores,
        gt_boxes,
        batch_ids,
        thresh=0.4
):
    # sort everything based on highest to lowest pred score
    pred_boxes, pred_scores, batch_ids = np.array(pred_boxes), np.array(pred_scores), np.array(batch_ids)
    AP = 0
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return AP

    thresh = 0.4
    index = pred_scores.argsort()[::-1]
    pred_scores = pred_scores[index]
    pred_boxes = pred_boxes[index]
    batch_ids = batch_ids[index]

    # initialize tp, all_p, gt, precision and recall
    true_pos, all_pos, all_gt = 0, 1e-7, len(gt_boxes)
    precision, recall = [], []

    # iterate over all predicted boxes
    matched = {}
    for i, pred_box in enumerate(pred_boxes):
        batch_id = batch_ids[i]  # get batch_id for current pred_box
        true_pos += is_true_positive(pred_box, gt_boxes, thresh, matched, batch_id)  # increment true pos
        all_pos += 1  # increment all_pos by 1 for each pred_box
        precision.append(true_pos / all_pos)  # append updated precision
        recall.append(true_pos / all_gt)  # append updated recall
    

    if len(precision) == 1:
        return precision[0]

    AP = sklearn.metrics.auc(recall, precision)

    return AP


def is_true_positive(pred_box, gt_boxes, thresh, matched, batch_id):
    max_iou = 0
    max_index = -1
    for i, gt_box in enumerate(gt_boxes):
        if i in matched or gt_box[4] != batch_id:
            continue
        curr_iou = iou(pred_box, gt_box[:-1])
        if curr_iou > max_iou:
            max_iou = curr_iou
            max_index = i

    if max_iou == 0:
        return 0

    matched[max_index] = 1
    if max_iou >= thresh:
        return 1
    return 0


# def compute_all_positives(cls_prob, bboxes, thresh):
#     nmsed_bboxes, nmsed_scores = nms(bboxes, probs, thresh)
#     return len(nmsed_bboxes), nmsed_bboxes, nmsed_scores


def test_net(model, val_loader=None, thresh=0.05, epoch=1):
    """
    tests the networks and visualize the detections
    thresh is the confidence threshold
    """

    global val_step
    losses = AverageMeter()

    pred_boxes = {i: [] for i in range(20)}
    pred_scores = {i: [] for i in range(20)}
    pred_classes = {i: [] for i in range(20)}
    batch_ids = {i: [] for i in range(20)}
    target_boxes = {i: [] for i in range(20)}

    for it, data in enumerate(val_loader):
        # one batch = data for one image
        image = data['image']
        target = data['label']
        wgt = data['wgt']
        rois = data['rois']
        gt_boxes = data['gt_boxes']
        gt_class_list = data['gt_classes']

        # TODO: perform forward pass, compute cls_probs
        image, rois, target = image.cuda(), rois.cuda(), target.cuda()
        cls_prob = model(image, rois * 512, target)
        loss = model.build_loss(cls_prob, target)

        losses.update(loss.item(), image.size(0))

        if it % disp_interval == 0:
            print('Test: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                it,
                len(val_loader),
                loss=losses))

        # TODO: Plot validation loss
        if USE_WANDB and it % 500 == 0:
            val_step += 1
            wandb.log({
                "Val/Loss": losses.avg,
                "Val/val_step": val_step,
            })

        # append batch_id to gt_boxes
        batch_id = torch.ones((len(gt_boxes), 1)) * it

        # import pdb
        # pdb.set_trace()
        gt_boxes = [torch.Tensor(gt_box) for gt_box in gt_boxes]
        gt_boxes = torch.stack(gt_boxes)
        gt_boxes = torch.cat((gt_boxes, batch_id), dim=-1)
        # TODO: Iterate over each class (follow comments)
        for class_idx in range(20):
            # get valid rois and cls_scores based on thresh
            probs = cls_prob[:, class_idx]
            nmsed_bboxes, nmsed_scores = nms(rois[0], probs, thresh)

            pred_boxes[class_idx].extend(nmsed_bboxes)
            pred_scores[class_idx].extend(nmsed_scores)
            pred_classes[class_idx].extend([class_idx] * len(nmsed_bboxes))
            batch_ids[class_idx].extend([it] * len(nmsed_bboxes))
            target_boxes[class_idx].extend([gt_box for i, gt_box in enumerate(gt_boxes) if gt_class_list[i].item() == class_idx])

            # TODO: visualize bounding box predictions when required
            # # visualize the first 10 images
            if USE_WANDB and it % 250 == 0:
                visualize_bbox(image, [class_idx] * len(nmsed_bboxes), nmsed_bboxes, nmsed_scores, epoch, it, eval="Val")

    # TODO: Calculate mAP on test set
    AP = [0] * 20
    for idx in range(20):
        AP[idx] = compute_ap(
            pred_boxes[idx],
            pred_scores[idx],
            target_boxes[idx],
            batch_ids[idx],
            thresh=0.5
        )

    # AP = (true_pos / all_pos) * (true_pos / gt_all)
    class_id_to_labels = dict(enumerate(val_dataset.CLASS_NAMES))

    for i, ap in enumerate(AP):
        if USE_WANDB:
            wandb.log({
                f'Val/AP/{class_id_to_labels[i]}' : ap,
            })

    if USE_WANDB:
        wandb.log({"Val/mAP": np.mean(AP)})
    return AP


def train(net, train_loader, val_loader, optimizer):
    # training
    train_loss = 0
    step_cnt = 0
    re_cnt = False
    disp_interval = 10
    val_interval = 1000
    num_epochs = 6

    global train_step

    losses = AverageMeter()

    for epoch in range(num_epochs):
        for it, data in enumerate(train_loader):


            # TODO: get one batch and perform forward pass
            # one batch = data for one image
            image = data['image']
            target = data['label']
            wgt = data['wgt']
            rois = data['rois']
            gt_boxes = data['gt_boxes']
            gt_class_list = data['gt_classes']

            # TODO: perform forward pass - take care that proposal values should be in pixels for the fwd pass
            # also convert inputs to cuda if training on GPU
            image, rois, target = image.cuda(), rois.cuda(), target.cuda()

            cls_prob = net(image, rois * 512, target)
            # backward pass and update
            loss = net.loss
            train_loss += loss.item()
            losses.update(loss.item(), image.size(0))


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # TODO: Plot training loss
            if USE_WANDB:
                train_step += 1
                wandb.log({
                    "Epoch": epoch,
                    "Train/Loss": losses.avg,
                    "Train/train_step": train_step,
                })
                # TODO: Perform all visualizations here
                # The intervals for different things are defined in the handout

            if (epoch == 0 or epoch == num_epochs - 1) and it % 250 == 0:
                scores, classes = cls_prob.max(dim=1)
                scores, index = scores.sort(descending=True)
                bboxes = rois[0, index]
                classes = classes[index]

                if USE_WANDB:
                    classes = classes.tolist()[:5]
                    bboxes = bboxes.tolist()[:5]
                    scores = scores.tolist()[:5]
                    visualize_bbox(image, classes, bboxes, scores, epoch, it)

        net.eval()
        ap = test_net(net, val_loader, epoch=epoch)
        print("AP ", ap)
        print("mAP", np.mean(ap))
        if USE_WANDB:
            # for label in range(20):
            #     wandb.log({f'class {label}': ap[label]})
            log_dict = {}
            for i in range(20):
                cls_name = VOCDataset.get_class_name(i)
                log_dict["Val/AP/{}".format(cls_name)] = ap[i]
                log_dict["Val/AP/Epoch"] = epoch
            wandb.log(log_dict)
            wandb.log({
                "Val/mAP": np.mean(ap), 
                "Val/epoch": epoch
                })
        net.train()

if __name__ == '__main__':
    main()
