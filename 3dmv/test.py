import argparse
import os
import shutil
import sys
from collections import OrderedDict

import numpy as np
import torch

import data_util
import util
from enet import create_enet_for_3d
from model import Model2d3d
from projection import ProjectionHelper

ENET_TYPES = {'scannet': (41, [0.496342, 0.466664, 0.440796], [0.277856, 0.28623, 0.291129])}  # classes, color mean/std

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--scene_list', required=True, help='path to file list of scenes to test')
parser.add_argument('--model_path', required=True, help='path to model')
parser.add_argument('--data_path_2d', required=True, help='path to 2d data')
parser.add_argument('--data_path_3d', required=True, help='path to 3d data')
parser.add_argument('--has_gt', action='store_true', help='test scenes have gt to evaluate against')
parser.add_argument('--output_path', default='./output', help='output path')
# Collab sometimes ends before the runtime, causing loss of all the models and logs file
parser.add_argument('--drive', default='', help='folder to upload model checkpoints on mounted google drive')
# test params
parser.add_argument('--num_classes', default=42, help='#classes')
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--num_nearest_images', type=int, required=True, help='#images')
parser.add_argument('--model2d_type', default='scannet', help='which enet (scannet)')
# parser.add_argument('--test_2d_model', dest='test_2d_model', action='store_true')
parser.add_argument('--model2d_orig_path', required=True, help='path to model')

# Network Architecture params - Scan completion and Use smaller model
parser.add_argument('--use_smaller_model', dest='use_smaller_model', action='store_true')
parser.add_argument('--test_scan_completion', dest='test_scan_completion', action='store_true',
                    help='test scan completion branch')

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

parser.set_defaults(train_2d_model=False)
opt = parser.parse_args()
assert opt.model2d_type in ENET_TYPES
print(opt)

# specify gpu
# os.environ['CUDA_VISIBLE_DEVICES']=str(opt.gpu)
CUDA_AVAILABLE = os.environ['CUDA_VISIBLE_DEVICES']

# create camera intrinsics
input_image_dims = [328, 256]
proj_image_dims = [41, 32]
intrinsic = util.make_intrinsic(opt.fx, opt.fy, opt.mx, opt.my)
intrinsic = util.adjust_intrinsic(intrinsic, [opt.intrinsic_image_width, opt.intrinsic_image_height], proj_image_dims)
if CUDA_AVAILABLE:
    intrinsic = intrinsic.cuda()

grid_dims = [opt.grid_dimX, opt.grid_dimY, opt.grid_dimZ]
column_height = opt.grid_dimZ
num_images = opt.num_nearest_images
grid_padX = opt.grid_dimX // 2
grid_padY = opt.grid_dimY // 2
color_mean = ENET_TYPES[opt.model2d_type][1]
color_std = ENET_TYPES[opt.model2d_type][2]

# TODO READ THIS FROM FILE INSTEAD OF HARDCODING
print('warning: using hard-coded scannet label set')
valid_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]

# create model
num_classes = opt.num_classes
projection = ProjectionHelper(intrinsic, opt.depth_min, opt.depth_max, proj_image_dims, grid_dims, opt.voxel_size)
model2d_fixed, model2d_trainable, model2d_classifier = create_enet_for_3d(ENET_TYPES[opt.model2d_type],
                                                                          opt.model2d_orig_path, num_classes)

if opt.test_scan_completion:
    model2dt_path = opt.model_path.replace('-model-semantic_and_scan.pth', '-model2d.pth')
    # model2d_trainable.load_state_dict(torch.load(model2dt_path, map_location='cpu'))
    model2d_trainable.load_state_dict(torch.load(model2dt_path))
else:
    model2dt_path = opt.model_path.replace('model.pth', 'model2d.pth')
    fixedname = os.path.basename(opt.model_path).split('model.pth')[0] + 'model2dfixed.pth'
    model2dfixed_path = os.path.join(os.path.dirname(opt.model_path), fixedname)
    model2d_fixed.load_state_dict(torch.load(model2dfixed_path))
    model2d_trainable.load_state_dict(torch.load(model2dt_path))

# if opt.test_2d_model:
#    model2dc_path = opt.model_path.replace('model.pth', 'model2dc.pth')
#    model2d_classifier.load_state_dict(torch.load(model2dc_path))
model = Model2d3d(num_classes, num_images, intrinsic, proj_image_dims, grid_dims,
                  opt.depth_min, opt.depth_max, opt.voxel_size, opt.use_smaller_model, opt.test_scan_completion)

# Fix Keys due to change in original network by Angela Dai
def change_model_keys_name(model_dict: OrderedDict, using_smaller_model):
    if using_smaller_model:
        model_key_mappings = {
            "classifier.0.weight": "semanticClassifier.0.weight",
            "classifier.0.bias": "semanticClassifier.0.bias",
            "classifier.3.weight": "semanticClassifier.2.weight",  # Smaller Model has no dropout layer
            "classifier.3.bias": "semanticClassifier.2.bias",
        }
    else:
        model_key_mappings = {
            "classifier.0.weight": "semanticClassifier.0.weight",
            "classifier.0.bias": "semanticClassifier.0.bias",
            "classifier.3.weight": "semanticClassifier.3.weight",
            "classifier.3.bias": "semanticClassifier.3.bias",
        }
    new_model_dict = OrderedDict()
    for k, v in model_dict.items():
        if k in model_key_mappings.keys():
            new_model_dict[model_key_mappings[k]] = v
        else:
            new_model_dict[k] = v
    return new_model_dict


model_dict = torch.load(opt.model_path, map_location=lambda storage, loc: storage)
model_dict = change_model_keys_name(model_dict, opt.use_smaller_model)
model.load_state_dict(model_dict)

print(model)

# move to gpu
if CUDA_AVAILABLE:
    print("Using GPU")
    model = model.cuda()
    model.eval()
    model2d_fixed = model2d_fixed.cuda()
    model2d_fixed.eval()
    model2d_trainable = model2d_trainable.cuda()
    model2d_trainable.eval()
else:
    model.eval()
    model2d_fixed.eval()
    model2d_trainable.eval()

# data files
scenes = util.read_lines_from_file(opt.scene_list)
print('#scenes = %d' % (len(scenes)))
if opt.has_gt:
    print('evaluating test scenes')
else:
    print('running model over test scenes (no evaluation)')

_SPLITTER = ','


def evaluate_prediction(scene_occ, scene_label, output):
    mask = np.equal(scene_occ[0], 1)
    output[np.logical_not(mask)] = 0
    mask = np.logical_and(mask, np.not_equal(scene_label, num_classes - 1))
    num_wrong = np.count_nonzero(scene_label.astype(np.int32)[mask] - output.astype(np.int32)[mask])
    inst_num_occ = np.count_nonzero(mask)
    inst_num_correct = inst_num_occ - num_wrong
    # class stats
    class_num_correct = np.zeros(num_classes)
    class_num_occ = np.zeros(num_classes)
    class_num_union = np.zeros(num_classes)
    for c in range(num_classes):
        if not c in valid_classes:
            continue
        mask = np.equal(scene_label, c)
        if np.any(mask):
            class_num_occ[c] = np.count_nonzero(mask)
            num_wrong = np.count_nonzero(scene_label.astype(np.int32)[mask] - output.astype(np.int32)[mask])
            class_num_correct[c] = class_num_occ[c] - num_wrong
        class_num_union[c] = np.count_nonzero(
            np.logical_or(mask, np.logical_and(np.not_equal(scene_label, num_classes - 1), np.equal(output, c))))

    print('instance acc = %f' % (float(inst_num_correct) / float(inst_num_occ)))
    class_acc = np.divide(class_num_correct, class_num_occ)
    class_iou = np.divide(class_num_correct, class_num_union)
    print('class_acc %f %f' % (np.nanmean(class_acc), class_acc))
    print('class_iou %f %f' % (np.nanmean(class_iou), class_iou))
    return {'instance_num_correct': inst_num_correct, 'instance_num_total': inst_num_occ,
            'class_num_correct': class_num_correct, 'class_num_total': class_num_occ,
            'class_num_union': class_num_union}


def test(scene_name, eval_file):
    print('scene %s' % (scene_name))
    scene_file = os.path.join(opt.data_path_3d, scene_name + '.sdf.ann')
    scene_image_file = os.path.join(opt.data_path_3d, scene_name + '.image')
    if not os.path.exists(scene_file) or not os.path.exists(scene_image_file):
        print(scene_file, os.path.exists(scene_file))
        print(scene_image_file, os.path.exists(scene_image_file))
        raise FileNotFoundError
    scene_occ, scene_label = data_util.load_scene(scene_file, num_classes, opt.has_gt)
    if scene_occ.shape[1] > column_height:
        scene_occ = scene_occ[:, :column_height, :, :]
        if opt.has_gt:
            scene_label = scene_label[:column_height, :, :]
    scene_occ_sz = scene_occ.shape[1:]
    depth_images, color_images, poses, frame_ids, world_to_grids = \
        data_util.load_scene_image_info_multi(scene_image_file, scene_name, opt.data_path_2d, proj_image_dims,
                                              input_image_dims, num_classes, color_mean, color_std)

    if CUDA_AVAILABLE:
        input_occ = torch.cuda.FloatTensor(1, 2, grid_dims[2], grid_dims[1], grid_dims[0])
        depth_image = torch.cuda.FloatTensor(num_images, proj_image_dims[1], proj_image_dims[0])
        color_image = torch.cuda.FloatTensor(num_images, 3, input_image_dims[1], input_image_dims[0])
        world_to_grid = torch.cuda.FloatTensor(num_images, 4, 4)
        pose = torch.cuda.FloatTensor(num_images, 4, 4)
    else:
        input_occ = torch.FloatTensor(1, 2, grid_dims[2], grid_dims[1], grid_dims[0])
        depth_image = torch.FloatTensor(num_images, proj_image_dims[1], proj_image_dims[0])
        color_image = torch.FloatTensor(num_images, 3, input_image_dims[1], input_image_dims[0])
        world_to_grid = torch.FloatTensor(num_images, 4, 4)
        pose = torch.FloatTensor(num_images, 4, 4)

    output_semantic_probs = np.zeros([num_classes, scene_occ_sz[0], scene_occ_sz[1], scene_occ_sz[2]])
    if opt.test_scan_completion:
        output_scan_probs = np.zeros([3, scene_occ_sz[0], scene_occ_sz[1], scene_occ_sz[2]])

    # Make sure poses and world_to_grid are Non-Singular
    for k in range(num_images):
        pose[k] = torch.eye(4)
        world_to_grid[k] = torch.eye(4)

    # go thru all columns
    for y in range(grid_padY, scene_occ_sz[1] - grid_padY):
        for x in range(grid_padX, scene_occ_sz[2] - grid_padX):
            input_occ.fill_(0)
            input_occ[0, :, :scene_occ_sz[0], :, :] = torch.from_numpy(scene_occ[:, :,
                                                                       y - grid_padY:y + grid_padY + 1,
                                                                       x - grid_padX:x + grid_padX + 1])
            cur_frame_ids = frame_ids[:, y, x][np.greater_equal(frame_ids[:, y, x], 0)]
            if len(cur_frame_ids) < num_images or torch.sum(input_occ[0, 0, :, grid_padY, grid_padX]) == 0:
                continue
            for k in range(num_images):
                depth_image[k] = depth_images[cur_frame_ids[k]]
                color_image[k] = color_images[cur_frame_ids[k]]
                pose[k] = poses[cur_frame_ids[k]]
                world_to_grid[k] = torch.from_numpy(world_to_grids[y, x])

            proj_mapping = [projection.compute_projection(d, c, t) for d, c, t in zip(depth_image, pose, world_to_grid)]
            if None in proj_mapping:
                print('Invalid sample at(x,y): (%3d, %3d)' % (x, y))
                continue

            proj_mapping = list(zip(*proj_mapping))
            proj_ind_3d = torch.stack(proj_mapping[0])
            proj_ind_2d = torch.stack(proj_mapping[1])
            imageft_fixed = model2d_fixed(torch.autograd.Variable(color_image))
            imageft = model2d_trainable(imageft_fixed)
            output_semantic, output_scan = model(torch.autograd.Variable(input_occ), imageft,
                                                 torch.autograd.Variable(proj_ind_3d),
                                                 torch.autograd.Variable(proj_ind_2d), grid_dims)

            # Take the output of max height of the grid
            output_semantic = output_semantic.data[0].permute(1, 0)  # (62, num_classes) => (num_classes, 62)
            output_semantic_probs[:, :, y, x] = output_semantic[:, :scene_occ_sz[0]]
            if opt.test_scan_completion:
                output_scan = output_scan.data[0].permute(1, 0)
                output_scan_probs[:, :, y, x] = output_scan[:, :scene_occ_sz[0]]  # Take the output of max height
        sys.stdout.write('\r[ %d | %d ]' % (y + 1, scene_occ_sz[1] - grid_padY))
        sys.stdout.flush()
    sys.stdout.write('\n')

    files_upload_names_list = []
    pred_semantic_label = np.argmax(output_semantic_probs, axis=0)

    # Compute the final predictions for all the known-occupied voxels and save to a file.
    all_known_voxels = np.equal(scene_occ[1], 1)
    pred_semantic_label_all_known = np.array(pred_semantic_label, copy=True)
    pred_semantic_label_all_known[np.logical_not(all_known_voxels)] = 0
    util.write_array_to_file(pred_semantic_label_all_known.astype(np.uint8),
                             os.path.join(opt.output_path, scene_name + '_semantic_AllKnownVoxels.bin'))
    files_upload_names_list.append(scene_name + '_semantic_AllKnownVoxels.bin')

    # Compute the final predictions, discard predictions for the unknown voxels and save to a file.
    mask_semantic_known_occupied = np.equal(scene_occ[0], 1)  # Only on voxels near the surface
    pred_semantic_label_all_known = np.array(pred_semantic_label, copy=True)  # Redundant copy
    pred_semantic_label_all_known[np.logical_not(mask_semantic_known_occupied)] = 0
    util.write_array_to_file(pred_semantic_label_all_known.astype(np.uint8),
                             os.path.join(opt.output_path, scene_name + '_semantic_KnownOccupiedOnly.bin'))
    files_upload_names_list.append(scene_name + '_semantic_KnownOccupiedOnly.bin')

    if opt.test_scan_completion:
        pred_scan_label = np.argmax(output_scan_probs, axis=0)
        # On Known-Free and Known-Occupied voxels. Discard Unknown.
        pred_scan_label[np.logical_not(all_known_voxels)] = 0
        filename = scene_name + '_scan.bin'
        util.write_array_to_file(pred_scan_label.astype(np.uint8), os.path.join(opt.output_path, filename))
        files_upload_names_list.append(filename)

    if opt.drive:
        for file_name in files_upload_names_list:  # Copy scene prediction files to google drive
            shutil.copyfile(os.path.join(opt.output_path, file_name), os.path.join(opt.drive, file_name))

    # ToDo: Implement evaluation for scan completion. Synthetic Data such as SunCG will be helpful here.
    eval_semantic_scene = None
    if opt.has_gt:
        eval_semantic_scene = evaluate_prediction(scene_occ, scene_label, pred_semantic_label)
    return eval_semantic_scene


def main():
    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)
    if opt.drive and not os.path.exists(opt.drive):
        os.makedirs(opt.drive)

    eval_file = None
    if opt.has_gt:
        eval_file = open(os.path.join(opt.output_path, 'eval.csv'), 'w')
        header_fields = ['scene']
        for c in valid_classes:
            header_fields.append('#corr class ' + str(c))
        for c in valid_classes:
            header_fields.append('#total class ' + str(c))
        for c in valid_classes:
            header_fields.append('#union class ' + str(c))
        header_fields.extend(['instance #corr', 'instance #total'])
        eval_file.write(_SPLITTER.join(header_fields) + '\n')

    # start testing
    inst_total_correct = 0
    inst_total_occ = 0
    class_total_correct = np.zeros(num_classes)
    class_total_occ = np.zeros(num_classes)
    class_total_union = np.zeros(num_classes)
    for scene in scenes:
        semantic_stats = test(scene, eval_file)
        if opt.has_gt:
            inst_total_correct += semantic_stats['instance_num_correct']
            inst_total_occ += semantic_stats['instance_num_total']
            class_total_correct += semantic_stats['class_num_correct']
            class_total_occ += semantic_stats['class_num_total']
            class_total_union += semantic_stats['class_num_union']
            fields = [scene]
            for c in valid_classes:
                fields.append(semantic_stats['class_num_correct'][c])
            for c in valid_classes:
                fields.append(semantic_stats['class_num_total'][c])
            for c in valid_classes:
                fields.append(semantic_stats['class_num_union'][c])
            fields.extend([semantic_stats['instance_num_correct'], semantic_stats['instance_num_total']])
            eval_file.write(_SPLITTER.join([str(f) for f in fields]) + '\n')

    if opt.has_gt:
        # summary stats
        instance_acc = float(inst_total_correct) / float(inst_total_occ)
        class_acc = np.divide(class_total_correct, class_total_occ)
        class_iou = np.divide(class_total_correct, class_total_union)
        # summary stats header
        fields = ['SUMMARY']
        for c in valid_classes:
            fields.append('class ' + str(c))
        eval_file.write(_SPLITTER.join([str(f) for f in fields]) + '\n')
        fields = ['%acc']
        for c in valid_classes:
            fields.append(class_acc[c])
        eval_file.write(_SPLITTER.join([str(f) for f in fields]) + '\n')
        fields = ['iou']
        for c in valid_classes:
            fields.append(class_iou[c])
        eval_file.write(_SPLITTER.join([str(f) for f in fields]) + '\n')
        fields = ['instance acc', str(instance_acc)]
        eval_file.write(_SPLITTER.join([str(f) for f in fields]) + '\n')
        eval_file.close()


if __name__ == '__main__':
    main()
