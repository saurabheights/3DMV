import argparse
import os
import random
import time

import numpy as np
import torch
import torchnet as tnt

import data_util
import util
from enet import create_enet_for_3d
from model import Model2d3d
from projection import ProjectionHelper
import shutil

ENET_TYPES = {'scannet': (41, [0.496342, 0.466664, 0.440796], [0.277856, 0.28623, 0.291129])}  # classes, color mean/std

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--train_data_list', required=True, help='path to file list of h5 train data')
parser.add_argument('--train_data_list_rootdir', required=True, help='path to root input_dir of paths in h5 train data filelist')
parser.add_argument('--val_data_list', default='', help='path to file list of h5 val data')
parser.add_argument('--output', default='./logs', help='folder to output model checkpoints')
# Collab sometimes ends before the runtime, causing loss of all the models and logs file
parser.add_argument('--drive', default='', help='folder to upload model checkpoints on mounted google drive')
parser.add_argument('--data_path_2d', required=True, help='path to 2d train data')
parser.add_argument('--class_weight_file', default='', help='path to histogram over classes')
# train params
parser.add_argument('--num_classes', default=42, help='#classes')
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
parser.add_argument('--max_epoch', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum, default=0.9')
parser.add_argument('--num_nearest_images', type=int, default=3, help='#images')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay, default=0.0005')
parser.add_argument('--retrain', dest='retrain', action='store_true', help='to retrain model')
parser.add_argument('--manualSeed', type=int, default=None, dest='manualSeed', help='Manual Seed for retraining')
parser.add_argument('--model_3d_path', default='', help='Path of 3d model')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
parser.add_argument('--model2d_type', default='scannet', help='which enet (scannet)')
parser.add_argument('--model2d_path', required=True, help='path to enet model')
parser.add_argument('--model2d_trainable_path', default='', help='Path of trainable part of 2d model')
parser.add_argument('--use_proxy_loss', dest='use_proxy_loss', action='store_true')
# Scan completion params
parser.add_argument('--use_smaller_model', dest='use_smaller_model', action='store_true')
parser.add_argument('--train_scan_completion', dest='train_scan_completion', action='store_true',
                    help='train scan completion branch')
parser.add_argument('--voxel_removal_fraction', dest='voxel_removal_fraction', default=0.5,
                    help='% of voxels to remove from center column')

# 2d/3d
parser.add_argument('--voxel_size', type=float, default=0.05, help='voxel size (in meters)')
parser.add_argument('--grid_dimX', type=int, default=31, help='3d grid dim x')
parser.add_argument('--grid_dimY', type=int, default=31, help='3d grid dim y')
parser.add_argument('--grid_dimZ', type=int, default=62, help='3d grid dim z')
parser.add_argument('--depth_min', type=float, default=0.4, help='min depth (in meters)')
parser.add_argument('--depth_max', type=float, default=4.0, help='max depth (in meters)')
# scannet intrinsic params
parser.add_argument('--intrinsic_image_width', type=int, default=640, help='2d image width')
parser.add_argument('--intrinsic_image_height', type=int, default=480, help='2d image height')
parser.add_argument('--fx', type=float, default=577.870605, help='intrinsics')
parser.add_argument('--fy', type=float, default=577.870605, help='intrinsics')
parser.add_argument('--mx', type=float, default=319.5, help='intrinsics')
parser.add_argument('--my', type=float, default=239.5, help='intrinsics')

parser.set_defaults(use_proxy_loss=False)
opt = parser.parse_args()
assert opt.model2d_type in ENET_TYPES
print(opt)
if opt.retrain:
    assert opt.model_3d_path and opt.model2d_trainable_path, \
        "No 3d model path and 2d trainable model path is provided, although retrain option is specified."

# specify gpu
# os.environ['CUDA_VISIBLE_DEVICES']=str(opt.gpu)
CUDA_AVAILABLE = os.environ['CUDA_VISIBLE_DEVICES']
if CUDA_AVAILABLE:
    print(torch.cuda.get_device_name(0))
    print('Using GPU')
    displayMemoryUsageOnce = True
else:
    print('Using CPU')
    displayMemoryUsageOnce = False

# Note: This may fail if random is called globally in  any project files which are imported above.
if opt.manualSeed is None:
    opt.manualSeed = random.randint(0, 99999)

print('Using Random Seed value as: %d' % opt.manualSeed)
torch.manual_seed(opt.manualSeed)  # Set for pytorch, used for cuda as well.
random.seed(opt.manualSeed)  # Set for python
np.random.seed(opt.manualSeed)  # Set for numpy

# create camera intrinsics
input_image_dims = [328, 256]
proj_image_dims = [41, 32]
intrinsic = util.make_intrinsic(opt.fx, opt.fy, opt.mx, opt.my)
intrinsic = util.adjust_intrinsic(intrinsic, [opt.intrinsic_image_width, opt.intrinsic_image_height], proj_image_dims)
if CUDA_AVAILABLE:
    intrinsic = intrinsic.cuda()
grid_dims = [opt.grid_dimX, opt.grid_dimY, opt.grid_dimZ]
column_height = opt.grid_dimZ
batch_size = opt.batch_size

assert batch_size <= 64, "Higher batch size will cause print statement to fail, search 64 // batch_size"
num_images = opt.num_nearest_images
grid_centerX = opt.grid_dimX // 2
grid_centerY = opt.grid_dimY // 2
color_mean = ENET_TYPES[opt.model2d_type][1]
color_std = ENET_TYPES[opt.model2d_type][2]

# Create 2d and 3d model
num_classes = opt.num_classes
model2d_fixed, model2d_trainable, model2d_classifier = create_enet_for_3d(ENET_TYPES[opt.model2d_type], opt.model2d_path, num_classes)
model = Model2d3d(num_classes, num_images, intrinsic, proj_image_dims, grid_dims, opt.depth_min, opt.depth_max, opt.voxel_size, opt.use_smaller_model, opt.train_scan_completion)

# Load model weights
if opt.retrain:
    model.load_state_dict(torch.load(opt.model_3d_path))
    model2d_trainable.load_state_dict(torch.load(opt.model2d_trainable_path))
    print("Loaded models from %s and %s" % (opt.model2d_trainable_path, opt.model_3d_path))

projection = ProjectionHelper(intrinsic, opt.depth_min, opt.depth_max, proj_image_dims, grid_dims, opt.voxel_size)

# Create criterion_weights for Semantic Segmentation
criterion_weights_semantic = torch.ones(num_classes)
if opt.class_weight_file:
    criterion_weights_semantic = util.read_class_weights_from_file(opt.class_weight_file, num_classes, True)
for c in range(num_classes):
    if criterion_weights_semantic[c] > 0:
        criterion_weights_semantic[c] = 1 / np.log(1.2 + criterion_weights_semantic[c])

print("Criterion Weights for semantic: \n%s" % criterion_weights_semantic.numpy())

# Create criterion_weights for Scan Completion
if opt.train_scan_completion:
    criterion_weights_scan = torch.zeros(3)
    criterion_weights_scan[0] = 1.0
    criterion_weights_scan[1] = 10.0
    criterion_weights_scan[2] = 0
    # Normalize as done for semantic. Keep a good balance between loss of both semantic and scan.
    criterion_weights_scan = criterion_weights_scan / torch.sum(criterion_weights_scan)
    for c in range(3):  # Not Used, What is 1.2 for?
        criterion_weights_scan[c] = 1 / np.log(1.2 + criterion_weights_scan[c])
    print("Criterion Weights for scan: \n%s" % criterion_weights_scan.numpy())

if CUDA_AVAILABLE:
    criterion_semantic = torch.nn.CrossEntropyLoss(criterion_weights_semantic).cuda()
    criterion2d = torch.nn.CrossEntropyLoss(criterion_weights_semantic).cuda()
else:
    criterion_semantic = torch.nn.CrossEntropyLoss(criterion_weights_semantic)
    criterion2d = torch.nn.CrossEntropyLoss(criterion_weights_semantic)

if opt.train_scan_completion:
    criterion_scan = torch.nn.CrossEntropyLoss(criterion_weights_scan)
    if CUDA_AVAILABLE:
        criterion_scan = criterion_scan.cuda()

# move to gpu
if CUDA_AVAILABLE:
    model2d_fixed = model2d_fixed.cuda()
    model2d_fixed.eval()
    model2d_trainable = model2d_trainable.cuda()
    model2d_classifier = model2d_classifier.cuda()
    model = model.cuda()
    print('Model moved to cuda')
else:
    model2d_fixed.eval()

optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
optimizer2d = torch.optim.SGD(model2d_trainable.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
if opt.use_proxy_loss:
    optimizer2dc = torch.optim.SGD(model2d_classifier.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

# data files
train_files = util.read_lines_from_file(opt.train_data_list)
train_files = [os.path.join(opt.train_data_list_rootdir, x) for x in train_files]  # Append root path to each filename
val_files = [] if not opt.val_data_list else util.read_lines_from_file(opt.val_data_list)
val_files = [os.path.join(opt.train_data_list_rootdir, x) for x in val_files]  # Append root path to each filename
print('#train files = %d' % (len(train_files)))
print('#val files = %d' % (len(val_files)))

# ToDo: Remove global variables. Causes more bug due to multiple variables with same name.
_NUM_OCCUPANCY_STATES = 3
_SPLITTER = ','
confusion = tnt.meter.ConfusionMeter(num_classes)
confusion2d = tnt.meter.ConfusionMeter(num_classes)
confusion_val = tnt.meter.ConfusionMeter(num_classes)
confusion2d_val = tnt.meter.ConfusionMeter(num_classes)
if opt.train_scan_completion:
    confusion_scan = tnt.meter.ConfusionMeter(_NUM_OCCUPANCY_STATES)
    confusion_scan_val = tnt.meter.ConfusionMeter(_NUM_OCCUPANCY_STATES)


def check_gpu_memory_usage_once():
    global displayMemoryUsageOnce
    if displayMemoryUsageOnce and CUDA_AVAILABLE:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        if device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')
        displayMemoryUsageOnce = False


def train(epoch, iter, log_file_semantic, log_file_scan, train_file, log_file_2d):
    train_loss_semantic = []  # To store semantic loss at each iteration
    train_loss_scan = []  # To store scan loss at each iteration
    train_loss_2d = []
    model.train()
    start = time.time()
    model2d_trainable.train()
    if opt.use_proxy_loss:
        model2d_classifier.train()

    # h5py has too much data. 10000 samples are too much to use. Divide by 10 and pick 1000 at a time
    print('Training on %s' % train_file)
    for h5py_index in range(10):
        volumes, labels, frames, world_to_grids = data_util.load_hdf5_data(train_file, num_classes, h5py_index)
        frames = frames[:, :2+num_images]
        volumes = volumes.permute(0, 1, 4, 3, 2)
        labels = labels.permute(0, 1, 4, 3, 2)
        labels = labels[:, 0, :, grid_centerX, grid_centerY]  # center columns as targets

        # Filter out the scenes not available
        available_frames_index = data_util.get_available_frames_id(opt.data_path_2d, frames)
        if len(available_frames_index) < batch_size:
            continue
        volumes = volumes[available_frames_index]
        labels = labels[available_frames_index]
        frames = frames[available_frames_index]
        world_to_grids = world_to_grids[available_frames_index]

        num_samples = volumes.shape[0]
        # shuffle
        indices = torch.randperm(num_samples).long().split(batch_size)
        # remove last mini-batch so that all the batches have equal size
        indices = indices[:-1]

        if CUDA_AVAILABLE:
            mask_semantic = torch.cuda.LongTensor(batch_size*column_height)
            depth_images = torch.cuda.FloatTensor(batch_size * num_images, proj_image_dims[1], proj_image_dims[0])
            color_images = torch.cuda.FloatTensor(batch_size * num_images, 3, input_image_dims[1], input_image_dims[0])
            camera_poses = torch.cuda.FloatTensor(batch_size * num_images, 4, 4)
            label_images = torch.cuda.LongTensor(batch_size * num_images, proj_image_dims[1], proj_image_dims[0])
        else:
            mask_semantic = torch.LongTensor(batch_size * column_height)
            depth_images = torch.FloatTensor(batch_size * num_images, proj_image_dims[1], proj_image_dims[0])
            color_images = torch.FloatTensor(batch_size * num_images, 3, input_image_dims[1], input_image_dims[0])
            camera_poses = torch.FloatTensor(batch_size * num_images, 4, 4)
            label_images = torch.LongTensor(batch_size * num_images, proj_image_dims[1], proj_image_dims[0])

        for t,v in enumerate(indices):
            iter_start = time.time()
            # print(t, v)
            if CUDA_AVAILABLE:
                targets_semantic = torch.autograd.Variable(labels[v].cuda())
            else:
                targets_semantic = torch.autograd.Variable(labels[v])

            # Ignore Invalid targets for semantic
            mask_semantic = targets_semantic.view(-1).data.clone()
            for k in range(num_classes):
                if criterion_weights_semantic[k] == 0:
                    mask_semantic[mask_semantic.eq(k)] = 0
            mask_semantic_indices = mask_semantic.nonzero().squeeze()  # Used in confusion matrix
            if len(mask_semantic_indices.shape) == 0:
                continue

            # Ignore Unknown Voxels for scan
            # occ[0] = np.less_equal(np.abs(sdfs), 1) # occupied space - 1, empty space - 0
            # occ[1] = np.greater_equal(sdfs, -1)     # known space = 1, unknown space - 0
            #                           occ0  occ1
            # Known-Free Space :           0,    1     (2) - Target = 0
            # Known-Occupied Space :       1,    1     (3) - Target = 1
            # Unknown Space:               0,    0     (0) - Target = 2
            # Create mask_scan from volume where 1 represents voxel is known-free or known-occupied.
            # 0 input should target 0, 1 should 1 and 2(from before voxel discarding) should 2.
            if opt.train_scan_completion:
                # Ignore Unknown Voxels from before.
                occ0 = volumes[v, 0, :, grid_centerX, grid_centerY].data.clone()
                occ1 = volumes[v, 1, :, grid_centerX, grid_centerY].data.clone()
                mask_scan = occ1.view(-1)  # Only Occupied Voxels
                mask_scan_indices = mask_scan.nonzero().squeeze()
                if len(mask_scan_indices.shape) == 0:
                    continue

                # ToDo: Some voxels are semantically labelled even though volumetric grid says they are Known-free.
                occ0 = occ0.view(-1)
                occ1 = occ1.view(-1)
                targets_scan = torch.LongTensor(batch_size * column_height)
                targets_scan[:] = 2  # Mark all as unknown
                targets_scan[torch.eq(occ0, 1) * torch.eq(occ1, 1)] = 1
                targets_scan[torch.eq(occ0, 0) * torch.eq(occ1, 1)] = 0
                if CUDA_AVAILABLE:
                    targets_scan = torch.autograd.Variable(targets_scan.cuda())
                else:
                    targets_scan = torch.autograd.Variable(targets_scan)

            transforms = world_to_grids[v].unsqueeze(1)
            transforms = transforms.expand(batch_size, num_images, 4, 4).contiguous().view(-1, 4, 4)
            if CUDA_AVAILABLE:
                transforms = transforms.cuda()

            # Load the 2d data
            is_load_success = data_util.load_frames_multi(opt.data_path_2d, frames[v], depth_images, color_images,
                                                          camera_poses, color_mean, color_std)
            if not is_load_success:
                continue

            # 3d Input
            volume = volumes[v].data.clone()
            # Get indices of voxels to be removed if training scan completion
            random_center_voxel_indices = torch.Tensor()  # Empty Tensor
            if opt.train_scan_completion:
                # Fixing Below Issues will improve training speed
                # ToDo: For all sample in each batch, same random voxels are removed.
                # ToDo: Voxel already unknown also gets removed.
                random_center_voxel_indices = projection.get_random_center_voxels_index(opt.voxel_removal_fraction)
                # Mark the 3D voxels as Unknown. For Unknown Voxels: Input is 0 but Target is 2.
                volume[:, :, random_center_voxel_indices, grid_centerX, grid_centerY] = 0

            # Compute projection mapping and mark center voxels as Unknown if training for scan completion
            proj_mapping = [projection.compute_projection(d, c, t,
                                                          random_center_voxel_indices)
                            for d, c, t in zip(depth_images, camera_poses, transforms)]
            if None in proj_mapping:  # Invalid sample
                print('No mapping in proj_mapping')
                continue
            proj_mapping = list(zip(*proj_mapping))
            proj_ind_3d = torch.stack(proj_mapping[0])
            proj_ind_2d = torch.stack(proj_mapping[1])

            if opt.use_proxy_loss:
                data_util.load_label_frames(opt.data_path_2d, frames[v], label_images, num_classes)
                mask2d = label_images.view(-1).clone()
                for k in range(num_classes):
                    if criterion_weights_semantic[k] == 0:
                        mask2d[mask2d.eq(k)] = 0
                mask2d = mask2d.nonzero().squeeze()
                if len(mask2d.shape) == 0:
                    continue  # nothing to optimize for here

            # 2d
            imageft_fixed = model2d_fixed(torch.autograd.Variable(color_images))
            imageft = model2d_trainable(imageft_fixed)
            if opt.use_proxy_loss:
                ft2d = model2d_classifier(imageft)
                ft2d = ft2d.permute(0, 2, 3, 1).contiguous()

            # 2d/3d
            input3d = torch.autograd.Variable(volume)
            if CUDA_AVAILABLE:
                input3d = input3d.cuda()

            # Forward Pass
            output_semantic, output_scan = model(input3d, imageft, torch.autograd.Variable(proj_ind_3d),
                                                 torch.autograd.Variable(proj_ind_2d), grid_dims)

            # Display Once GPU memory usage - Be Sure of GPU usage. Collab is a bit unpredictable
            check_gpu_memory_usage_once()

            # Compute Scan and semantic Loss
            loss_semantic = criterion_semantic(output_semantic.view(-1, num_classes), targets_semantic.view(-1))
            train_loss_semantic.append(loss_semantic.item())
            if opt.train_scan_completion:
                loss_scan = criterion_scan(output_scan.view(-1, _NUM_OCCUPANCY_STATES),
                                           targets_scan.view(-1))
                train_loss_scan.append(loss_scan.item())
                loss = loss_scan + loss_semantic
            else:
                loss = loss_semantic

            # Backpropagate total loss.
            # ToDo: Note using same optimizer for both branches. Is there a need for different optimizers?
            optimizer.zero_grad()
            optimizer2d.zero_grad()
            if opt.use_proxy_loss:
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            optimizer.step()
            # optimizer2d.step is probably required even when use_proxy_loss is False, since backprojection layer is
            # differentiable, allowing us to backpropagate the gradients to 2d model from model(3D).
            optimizer2d.step()

            # ToDo: Check if proxy loss is required. If optimizer2d is injecting gradients, proxy loss may be needed.
            if opt.use_proxy_loss:
                loss2d = criterion2d(ft2d.view(-1, num_classes), torch.autograd.Variable(label_images.view(-1)))
                train_loss_2d.append(loss2d.item())
                optimizer2d.zero_grad()
                optimizer2dc.zero_grad()
                loss2d.backward()
                optimizer2dc.step()
                optimizer2d.step()
                # confusion
                y = ft2d.data
                y = y.view(-1, num_classes)[:, :-1]
                _, predictions = y.max(1)
                predictions = predictions.view(-1)
                k = label_images.view(-1)
                confusion2d.add(torch.index_select(predictions, 0, mask2d), torch.index_select(k, 0, mask2d))

            # Confusion for Semantic
            y = output_semantic.data
            # Discard semantic prediction of class num_classes-1[Unknown Voxel]
            y = y.view(y.nelement() // y.size(2), num_classes)[:, :-1]
            _, predictions = y.max(1)
            predictions = predictions.view(-1)
            k = targets_semantic.data.view(-1)
            confusion.add(torch.index_select(predictions, 0, mask_semantic_indices),
                          torch.index_select(k, 0, mask_semantic_indices))

            # Confusion for Scan completion
            if opt.train_scan_completion:
                y = output_scan.data
                # Discard semantic prediction of Unknown Voxels in target_scan
                y = y.view(y.nelement() // y.size(2), _NUM_OCCUPANCY_STATES)[:, :-1]
                _, predictions_scan = y.max(1)
                predictions_scan = predictions_scan.view(-1)
                k = targets_scan.data.view(-1)
                confusion_scan.add(torch.index_select(predictions_scan, 0, mask_scan_indices),
                                   torch.index_select(k, 0, mask_scan_indices))

            # Log loss for current iteration and print every 20th turn.
            msg1 = _SPLITTER.join([str(f) for f in [epoch, iter, loss_semantic.item()]])
            log_file_semantic.write(msg1 + '\n')
            if opt.train_scan_completion:
                msg2 = _SPLITTER.join([str(f) for f in [epoch, iter, loss_scan.item()]])
                log_file_scan.write(msg2 + '\n')

            # InFrequent logging stops chrome from crash[Colab] and also less strain on jupyter.
            if iter % (64 // batch_size) == 0:
                print("Semantic: %s, %0.6f" % (msg1, time.time() - iter_start))
                if opt.train_scan_completion:
                    print("Scan    : %s" % msg2)

            iter += 1
            if iter % (10000//batch_size) == 0:  # Save more frequently, since its Google Collaboratory.
                # Save 3d model
                if not opt.train_scan_completion:
                    torch.save(model.state_dict(),
                               os.path.join(opt.output, 'model-semantic-epoch%s-iter%s-Sem%s.pth'
                                            % (epoch, iter, str(loss_semantic.item()))))
                else:
                    torch.save(model.state_dict(),
                               os.path.join(opt.output, 'model-semantic_and_scan-epoch%s-iter%s-sem%s-scan%s.pth'
                                            % (epoch, iter, str(loss_semantic.item()), str(loss_scan.item()))))
                # Save 2d model
                # Important ToDo: Do we need to retrain on model2d_trainable
                torch.save(model2d_trainable.state_dict(),
                           os.path.join(opt.output, 'model2d-iter%s-epoch%s.pth' % (iter, epoch)))
                if opt.use_proxy_loss:
                    torch.save(model2d_classifier.state_dict(),
                               os.path.join(opt.output, 'model2dc-iter%s-epoch%s.pth' % (iter, epoch)))
            if iter == 1:
                torch.save(model2d_fixed.state_dict(), os.path.join(opt.output, 'model2dfixed.pth'))

            if iter % 100 == 0:
                evaluate_confusion(confusion, train_loss_semantic, epoch, iter, -1, 'TrainSemantic', log_file_semantic, num_classes)
                if opt.train_scan_completion:
                    evaluate_confusion(confusion_scan, train_loss_scan, epoch, iter, -1, 'TrainScan', log_file_scan, _NUM_OCCUPANCY_STATES)
                if opt.use_proxy_loss:
                    evaluate_confusion(confusion2d, train_loss_2d, epoch, iter, -1, 'Train2d', log_file_2d, num_classes)

    end = time.time()
    took = end - start
    evaluate_confusion(confusion, train_loss_semantic, epoch, iter, took, 'TrainSemantic', log_file_semantic, num_classes)
    if opt.train_scan_completion:
        evaluate_confusion(confusion_scan, train_loss_scan, epoch, iter, took, 'TrainScan', log_file_scan, _NUM_OCCUPANCY_STATES)
    if opt.use_proxy_loss:
        evaluate_confusion(confusion2d, train_loss_2d, epoch, iter, took, 'Train2d', log_file_2d, num_classes)
    return train_loss_semantic, train_loss_scan, iter, train_loss_2d




def test(epoch, iter, log_file_semantic_val, log_file_scan_val, val_file, log_file_2d_val):
    test_loss_semantic = []  # To store semantic loss at each iteration
    test_loss_scan = []  # To store scan loss at each iteration
    test_loss_2d = []
    model.eval()
    model2d_fixed.eval()
    model2d_trainable.eval()
    if opt.use_proxy_loss:
        model2d_classifier.eval()
    start = time.time()

    # h5py has too much data. 10000 samples are too much to use. Divide by 10 and pick 1000 at a time
    print('Validating on %s' % val_file)
    for h5py_index in range(10):
        volumes, labels, frames, world_to_grids = data_util.load_hdf5_data(val_file, num_classes, h5py_index)
        frames = frames[:, :2+num_images]
        volumes = volumes.permute(0, 1, 4, 3, 2)
        labels = labels.permute(0, 1, 4, 3, 2)
        labels = labels[:, 0, :, grid_centerX, grid_centerY]  # center columns as targets

        # Filter out the scenes not available
        available_frames_index = data_util.get_available_frames_id(opt.data_path_2d, frames)
        if len(available_frames_index) < batch_size:
            continue
        volumes = volumes[available_frames_index]
        labels = labels[available_frames_index]
        frames = frames[available_frames_index]
        world_to_grids = world_to_grids[available_frames_index]

        num_samples = volumes.shape[0]
        # shuffle
        indices = torch.randperm(num_samples).long().split(batch_size)
        # remove last mini-batch so that all the batches have equal size
        indices = indices[:-1]

        with torch.no_grad():
            if CUDA_AVAILABLE:
                mask_semantic = torch.cuda.LongTensor(batch_size*column_height)
                depth_images = torch.cuda.FloatTensor(batch_size * num_images, proj_image_dims[1], proj_image_dims[0])
                color_images = torch.cuda.FloatTensor(batch_size * num_images, 3, input_image_dims[1], input_image_dims[0])
                camera_poses = torch.cuda.FloatTensor(batch_size * num_images, 4, 4)
                label_images = torch.cuda.LongTensor(batch_size * num_images, proj_image_dims[1], proj_image_dims[0])
            else:
                mask_semantic = torch.LongTensor(batch_size*column_height)
                depth_images = torch.FloatTensor(batch_size * num_images, proj_image_dims[1], proj_image_dims[0])
                color_images = torch.FloatTensor(batch_size * num_images, 3, input_image_dims[1], input_image_dims[0])
                camera_poses = torch.FloatTensor(batch_size * num_images, 4, 4)
                label_images = torch.LongTensor(batch_size * num_images, proj_image_dims[1], proj_image_dims[0])

            for t,v in enumerate(indices):
                # print(t, v)
                if CUDA_AVAILABLE:
                    targets_semantic = labels[v].cuda()
                else:
                    targets_semantic = labels[v]

                # Ignore Invalid targets for semantic
                mask_semantic = targets_semantic.view(-1).data.clone()
                for k in range(num_classes):
                    if criterion_weights_semantic[k] == 0:
                        mask_semantic[mask_semantic.eq(k)] = 0
                mask_indices_semantic = mask_semantic.nonzero().squeeze()
                if len(mask_indices_semantic.shape) == 0:
                    continue

                # Ignore Unknown Voxels for scan
                # occ[0] = np.less_equal(np.abs(sdfs), 1) # occupied space - 1, empty space - 0
                # occ[1] = np.greater_equal(sdfs, -1)     # known space = 1, unknown space - 0
                #                           occ0  occ1
                # Known-Free Space :           0,    1     (2) - Target = 0
                # Known-Occupied Space :       1,    1     (3) - Target = 1
                # Unknown Space:               0,    0     (0) - Target = 2
                # Create mask_scan from volume where 1 represents voxel is known-free or known-occupied.
                # 0 input should target 0, 1 should 1 and 2(from before voxel discarding) should 2.
                if opt.train_scan_completion:
                    # Ignore Unknown Voxels from before.
                    occ0 = volumes[v, 0, :, grid_centerX, grid_centerY].data.clone()
                    occ1 = volumes[v, 1, :, grid_centerX, grid_centerY].data.clone()
                    mask_scan = occ1.view(-1)  # Only Occupied Voxels
                    mask_scan_indices = mask_scan.nonzero().squeeze()
                    if len(mask_scan_indices.shape) == 0:
                        continue

                    # ToDo: Some voxels are semantically labelled even though volumetric grid says they are Known-free.
                    occ0 = occ0.view(-1)
                    occ1 = occ1.view(-1)
                    targets_scan = torch.LongTensor(batch_size*column_height)
                    targets_scan[:] = 2  # Mark all as unknown
                    targets_scan[torch.eq(occ0, 1) * torch.eq(occ1, 1)] = 1
                    targets_scan[torch.eq(occ0, 0) * torch.eq(occ1, 1)] = 0
                    if CUDA_AVAILABLE:
                        targets_scan = torch.autograd.Variable(targets_scan.cuda())
                    else:
                        targets_scan = torch.autograd.Variable(targets_scan)

                transforms = world_to_grids[v].unsqueeze(1)
                transforms = transforms.expand(batch_size, num_images, 4, 4).contiguous().view(-1, 4, 4)
                if CUDA_AVAILABLE:
                    transforms = transforms.cuda()

                # Load the 2d data
                is_load_success = data_util.load_frames_multi(opt.data_path_2d, frames[v], depth_images,
                                                              color_images, camera_poses, color_mean, color_std)
                if not is_load_success:
                    continue

                # 3d Input
                volume = volumes[v].data.clone()
                # Get indices of voxels to be removed if training scan completion
                random_center_voxel_indices = torch.Tensor()  # Empty Tensor
                if opt.train_scan_completion:
                    # Fixing Below Issues will improve training speed
                    # ToDo: For all sample in each batch, same random voxels are removed.
                    # ToDo: Voxel already unknown also gets removed.
                    random_center_voxel_indices = projection.get_random_center_voxels_index(
                        opt.voxel_removal_fraction)
                    # Mark the 3D voxels as Unknown. For Unknown Voxels: Input is 0 but Target is 2
                    volume[:, :, random_center_voxel_indices, grid_centerX, grid_centerY] = 0

                # Compute projection mapping and mark center voxels as Unknown if training for scan completion
                proj_mapping = [projection.compute_projection(d, c, t, random_center_voxel_indices)
                                for d, c, t in zip(depth_images, camera_poses, transforms)]
                if None in proj_mapping:
                    print('No mapping in proj_mapping')
                    continue
                proj_mapping = list(zip(*proj_mapping))
                proj_ind_3d = torch.stack(proj_mapping[0])
                proj_ind_2d = torch.stack(proj_mapping[1])

                if opt.use_proxy_loss:
                    data_util.load_label_frames(opt.data_path_2d, frames[v], label_images, num_classes)
                    mask2d = label_images.view(-1).clone()
                    for k in range(num_classes):
                        if criterion_weights_semantic[k] == 0:
                            mask2d[mask2d.eq(k)] = 0
                    mask2d = mask2d.nonzero().squeeze()
                    if len(mask2d.shape) == 0:
                        continue  # nothing to optimize for here

                # 2d
                imageft_fixed = model2d_fixed(color_images)
                imageft = model2d_trainable(imageft_fixed)
                if opt.use_proxy_loss:
                    ft2d = model2d_classifier(imageft)
                    ft2d = ft2d.permute(0, 2, 3, 1).contiguous()

                # 2d/3d
                if CUDA_AVAILABLE:
                    input3d = volume.cuda()
                else:
                    input3d = volume

                # Forward Pass Only
                output_semantic, output_scan = model(input3d, imageft, proj_ind_3d, proj_ind_2d, grid_dims)

                # Compute Scan and semantic Loss
                loss_semantic = criterion_semantic(output_semantic.view(-1, num_classes), targets_semantic.view(-1))
                test_loss_semantic.append(loss_semantic.item())
                if opt.train_scan_completion:
                    loss_scan = criterion_scan(output_scan.view(-1, _NUM_OCCUPANCY_STATES),
                                               targets_scan.view(-1))
                    test_loss_scan.append(loss_scan.item())

                if opt.use_proxy_loss:
                    loss2d = criterion2d(ft2d.view(-1, num_classes), label_images.view(-1))
                    test_loss_2d.append(loss2d.item())
                    # Confusion
                    y = ft2d.data
                    y = y.view(-1, num_classes)[:, :-1]
                    _, predictions = y.max(1)
                    predictions = predictions.view(-1)
                    k = label_images.view(-1)
                    confusion2d_val.add(torch.index_select(predictions, 0, mask2d), torch.index_select(k, 0, mask2d))

                # Confusion for Semantic
                y = output_semantic.data
                y = y.view(y.nelement() // y.size(2), num_classes)[:, :-1]
                _, predictions = y.max(1)
                predictions = predictions.view(-1)
                k = targets_semantic.data.view(-1)
                confusion_val.add(torch.index_select(predictions, 0, mask_indices_semantic),
                                  torch.index_select(k, 0, mask_indices_semantic))

                # Confusion for Scan completion
                if opt.train_scan_completion:
                    y = output_scan.data
                    # Discard Scan prediction of Unknown Voxels in target_scan
                    y = y.view(y.nelement() // y.size(2), _NUM_OCCUPANCY_STATES)[:, :-1]
                    _, predictions_scan = y.max(1)
                    predictions_scan = predictions_scan.view(-1)
                    k = targets_scan.data.view(-1)
                    confusion_scan_val.add(torch.index_select(predictions_scan, 0, mask_scan_indices),
                                           torch.index_select(k, 0, mask_scan_indices))

    end = time.time()
    took = end - start
    evaluate_confusion(confusion_val, test_loss_semantic, epoch, iter, took,
                       'ValidationSemantic', log_file_semantic_val, num_classes)
    if opt.train_scan_completion:
        evaluate_confusion(confusion_scan_val, test_loss_scan, epoch, iter, took,
                           'ValidationScan', log_file_scan_val, _NUM_OCCUPANCY_STATES)
    if opt.use_proxy_loss:
        evaluate_confusion(confusion2d_val, test_loss_2d, epoch, iter, took, 'Validation2d', log_file_2d_val, num_classes)
    return test_loss_semantic, test_loss_scan, test_loss_2d


def evaluate_confusion(confusion_matrix, loss, epoch, iter, time, which, log_file, _num_classes):
    conf = confusion_matrix.value()
    total_correct = 0
    valids = np.zeros(_num_classes, dtype=np.float32)
    for c in range(_num_classes):
        num = conf[c,:].sum()
        valids[c] = -1 if num == 0 else float(conf[c][c]) / float(num)
        total_correct += conf[c][c]
    instance_acc = -1 if conf.sum() == 0 else float(total_correct) / float(conf.sum())
    avg_acc = -1 if np.all(np.equal(valids, -1)) else np.mean(valids[np.not_equal(valids, -1)])
    loss_mean = torch.mean(torch.Tensor(loss))
    log_file.write(_SPLITTER.join([str(f) for f in [epoch, iter, loss_mean.item(), avg_acc, instance_acc, time]]) + '\n')
    log_file.flush()

    print('Epoch: {}\tIter: {}\tLoss: {:.6f}\tAcc(inst): {:.6f}\tAcc(avg): {:.6f}\tTook: {:.2f}\t{}'.format(
        epoch, iter, loss_mean.data, instance_acc, avg_acc, time, which))


def main():
    if not os.path.exists(opt.output):
        os.makedirs(opt.output)
    if opt.drive and not os.path.exists(opt.drive):
        os.makedirs(opt.drive)

    # Log files to upload to google drive as well at the end of each epoch
    files_upload_names_list = list()

    # ToDo: Reduce the below log files code as done for log_file.close at the end.
    log_file_semantic = open(os.path.join(opt.output, 'log_semantic_train.csv'), 'w')
    files_upload_names_list.append('log_semantic_train.csv')
    log_file_semantic.write(_SPLITTER.join(['epoch', 'iter', 'loss', 'avg acc', 'instance acc', 'time']) + '\n')
    log_file_semantic.flush()

    log_file_scan = None
    if opt.train_scan_completion:
        log_file_scan = open(os.path.join(opt.output, 'log_scan_train.csv'), 'w')
        files_upload_names_list.append('log_scan_train.csv')
        log_file_scan.write(_SPLITTER.join(['epoch', 'iter', 'loss', 'avg acc', 'instance acc', 'time']) + '\n')
        log_file_scan.flush()

    log_file_2d = None
    if opt.use_proxy_loss:
        log_file_2d = open(os.path.join(opt.output, 'log2d_train.csv'), 'w')
        files_upload_names_list.append('log2d_train.csv')
        log_file_2d.write(_SPLITTER.join(['epoch', 'iter', 'loss', 'avg acc', 'instance acc', 'time']) + '\n')
        log_file_2d.flush()

    has_val = len(val_files) > 0
    log_file_semantic_val = None
    log_file_scan_val = None
    log_file_2d_val = None
    if has_val:
        log_file_semantic_val = open(os.path.join(opt.output, 'log_semantic_val.csv'), 'w')
        files_upload_names_list.append('log_semantic_val.csv')
        log_file_semantic_val.write(_SPLITTER.join(['epoch', 'iter', 'loss', 'avg acc', 'instance acc', 'time']) + '\n')
        log_file_semantic_val.flush()

        if opt.train_scan_completion:
            log_file_scan_val = open(os.path.join(opt.output, 'log_scan_val.csv'), 'w')
            files_upload_names_list.append('log_scan_val.csv')
            log_file_scan_val.write(_SPLITTER.join(['epoch', 'iter', 'loss', 'avg acc', 'instance acc', 'time']) + '\n')
            log_file_scan_val.flush()

        if opt.use_proxy_loss:
            log_file_2d_val = open(os.path.join(opt.output, 'log2d_val.csv'), 'w')
            files_upload_names_list.append('log2d_val.csv')
            log_file_2d_val.write(_SPLITTER.join(['epoch', 'iter', 'loss', 'avg acc', 'instance acc', 'time']) + '\n')
            log_file_2d_val.flush()

    if not opt.drive:  # Remove all elements in files_upload_src_dst_map if drive folder is not provided.
        files_upload_names_list.clear()

    # Start training
    print('Starting Training...')
    iter = 0
    # Note: In 3dmv, validation is done on gap of training on 10 files which is 1/10.
    num_files_per_val = max(round(len(train_files) / 2), 1)
    for epoch in range(opt.start_epoch, opt.start_epoch+opt.max_epoch):
        train_semantic_loss = []
        train_scan_loss = []
        train2d_loss = []
        val_semantic_loss = []
        val_scan_loss = []
        val2d_loss = []
        # Process shuffled train files
        train_file_indices = torch.randperm(len(train_files))
        for k in range(len(train_file_indices)):
            print('Epoch: {}\tFile: {}/{}\t{}'.format(epoch, k, len(train_files), train_files[train_file_indices[k]]))

            # Train
            loss_semantic, loss_scan, iter, loss2d = \
                train(epoch, iter, log_file_semantic, log_file_scan, train_files[train_file_indices[k]], log_file_2d)

            # Save all losses
            train_semantic_loss.extend(loss_semantic)
            if opt.train_scan_completion:
                train_scan_loss.extend(loss_scan)
            if loss2d:
                train2d_loss.extend(loss2d)

            # Validation
            if has_val and k % num_files_per_val == 0:
                val_index = torch.randperm(len(val_files))[0]
                loss_semantic, loss_scan, loss2d = \
                    test(epoch, iter, log_file_semantic_val, log_file_scan_val, val_files[val_index], log_file_2d_val)

                val_semantic_loss.extend(loss_semantic)
                if opt.train_scan_completion:
                    val_scan_loss.extend(loss_scan)
                if loss2d:
                    val2d_loss.extend(loss2d)

        evaluate_confusion(confusion, train_semantic_loss, epoch, iter, -1, 'TrainSemantic', log_file_semantic, num_classes)
        if opt.train_scan_completion:
            evaluate_confusion(confusion_scan, train_scan_loss, epoch, iter, -1, 'TrainScan', log_file_scan, _NUM_OCCUPANCY_STATES)
        if opt.use_proxy_loss:
            evaluate_confusion(confusion2d, train2d_loss, epoch, iter, -1, 'Train2d', log_file_2d, num_classes)

        for file_name in files_upload_names_list:  # Copy log files to google drive
            shutil.copyfile(os.path.join(opt.output, file_name),
                            os.path.join(opt.drive, "epoch-%s-%s" % (epoch, file_name)))
        if has_val:
            evaluate_confusion(confusion_val, val_semantic_loss, epoch, iter, -1,
                               'ValidationSemantic', log_file_semantic_val, num_classes)
            if opt.train_scan_completion:
                evaluate_confusion(confusion_scan_val, val_scan_loss, epoch, iter, -1, 'ValidationScan', log_file_scan_val, _NUM_OCCUPANCY_STATES)
            if opt.use_proxy_loss:
                evaluate_confusion(confusion2d_val, val2d_loss, epoch, iter, -1, 'Validation2d', log_file_2d_val, num_classes)

        if not opt.train_scan_completion:
            torch.save(model.state_dict(), os.path.join(opt.output, 'epoch-%s-model-semantic.pth' % epoch))
            if opt.drive:
                torch.save(model.state_dict(), os.path.join(opt.drive, 'epoch-%s-model-semantic_and_scan.pth' % epoch))
        else:
            torch.save(model.state_dict(), os.path.join(opt.output, 'epoch-%s-model-semantic_and_scan.pth' % epoch))
            if opt.drive:
                torch.save(model.state_dict(), os.path.join(opt.drive, 'epoch-%s-model-semantic_and_scan.pth' % epoch))

        torch.save(model2d_trainable.state_dict(), os.path.join(opt.output, 'epoch-%s-model2d.pth' % epoch))
        if opt.drive:
            torch.save(model2d_trainable.state_dict(), os.path.join(opt.drive, 'epoch-%s-model2d.pth' % epoch))

        if opt.use_proxy_loss:
            torch.save(model2d_classifier.state_dict(), os.path.join(opt.output, 'epoch-%s-model2dc.pth' % epoch))
            if opt.drive:
                torch.save(model2d_classifier.state_dict(), os.path.join(opt.drive, 'epoch-%s-model2dc.pth' % epoch))
        confusion.reset()
        confusion_val.reset()
        confusion_scan.reset()
        confusion_scan_val.reset()
        confusion2d.reset()
        confusion2d_val.reset()

    # Close all log files
    log_files = [log_file_semantic, log_file_semantic_val, log_file_scan, log_file_scan_val, log_file_2d, log_file_2d_val]
    log_files = list(filter(lambda x: x is not None, log_files))  # Remove None
    list(map(lambda f: f.close(), log_files))


if __name__ == '__main__':
    main()
