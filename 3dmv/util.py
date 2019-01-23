import os, struct, math

import matplotlib
import numpy as np
import torch

import itertools
from matplotlib import pyplot as plt

matplotlib.use('Agg')

semantic_classes = ['Free', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floor mat', 'clothes', 'ceiling', 'books', 'refridgerator', 'television', 'paper', 'towel', 'shower curtain', 'box', 'whiteboard', 'person', 'nightstand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop', 'Ignored']

scan_classes = ['Free', 'Known-Occupied', 'Unknown']

def plot_confusion_matrix(cm,
                          title='Confusion matrix',
                          normalize=True,
                          image_path = None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm.astype('float') / cm.sum()
        print("Normalized " + title)
    else:
        print(title + ', without normalization')

    # Plot non-normalized confusion matrix
    if (cm.shape[0] < 10):
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        if 'scan' in title.lower():
            classes = scan_classes
        else:
            classes = semantic_classes
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    confusion_matrix_str = '['
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if j == 0:
            confusion_matrix_str += '\n['
        confusion_matrix_str += plt.text(j, i, format(cm[i, j], fmt),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")._text
        if j<=cm.shape[1]-1:
            confusion_matrix_str+=', '
        else:
            confusion_matrix_str += '],'


    confusion_matrix_str += ']\n'

    if (cm.shape[0] < 10):  # Too Many classes wont work
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(image_path)

    return confusion_matrix_str


# util for saving tensors, for debug purposes
def write_array_to_file(tensor, filename):
    sz = tensor.shape
    with open(filename, 'wb') as f:
        f.write(struct.pack('Q', sz[0]))
        f.write(struct.pack('Q', sz[1]))
        f.write(struct.pack('Q', sz[2]))
        tensor.tofile(f)


def read_lines_from_file(filename):
    assert os.path.isfile(filename), filename
    lines = open(filename).read().splitlines()
    return lines


# create camera intrinsics
def make_intrinsic(fx, fy, mx, my):
    intrinsic = torch.eye(4)
    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = mx
    intrinsic[1][2] = my
    return intrinsic


# create camera intrinsics
def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic


def get_sample_files(samples_path):
    files = [f for f in os.listdir(samples_path) if f.endswith('.sample')]  # and os.path.isfile(join(samples_path, f))]
    return files


def get_sample_files_for_scene(scene, samples_path):
    files = [f for f in os.listdir(samples_path) if
             f.startswith(scene) and f.endswith('.sample')]  # and os.path.isfile(join(samples_path, f))]
    print('found %d for %s' % (len(files), os.path.join(samples_path, scene)))
    return files


def load_pose(filename):
    assert os.path.isfile(filename)
    lines = open(filename).read().splitlines()
    assert len(lines) == 4
    lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
    return torch.from_numpy(np.asarray(lines).astype(np.float32))


def read_class_weights_from_file(filename, num_classes, normalize):
    assert os.path.isfile(filename)
    weights = torch.zeros(num_classes)
    lines = open(filename).read().splitlines()
    for line in lines:
        parts = line.split('\t')
        assert len(parts) == 2
        weights[int(parts[0])] = int(parts[1])
    if normalize:
        weights = weights / torch.sum(weights)
    return weights
