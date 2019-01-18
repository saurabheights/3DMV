"""
This file provides method to convert Volumetric Grid Dumped By util.write_array_to_file
"""
import argparse
import os
import struct
import math
from os import listdir
from os.path import join, isfile

import numpy as np
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default='./logs/TestingModels/', help='Folder containing .bin output files from testing')
parser.add_argument('--output_path', default='./logs/TestingModels/', help='Path to output folder')
opt = parser.parse_args()
print(opt)

sys.path.append('.')

ID_COLOR = {
    0: [0, 0, 0],
    1: [174, 199, 232],
    2: [152, 223, 138],
    3: [31, 119, 180],
    4: [255, 187, 120],
    5: [188, 189, 34],
    6: [140, 86, 75],
    7: [255, 152, 150],
    8: [214, 39, 40],
    9: [197, 176, 213],
    10: [148, 103, 189],
    11: [196, 156, 148],
    12: [23, 190, 207],
    13: [178, 76, 76],
    14: [247, 182, 210],
    15: [66, 188, 102],
    16: [219, 219, 141],
    17: [140, 57, 197],
    18: [202, 185, 52],
    19: [51, 176, 203],
    20: [200, 54, 131],
    21: [92, 193, 61],
    22: [78, 71, 183],
    23: [172, 114, 82],
    24: [255, 127, 14],
    25: [91, 163, 138],
    26: [153, 98, 156],
    27: [140, 153, 101],
    28: [158, 218, 229],
    29: [100, 125, 154],
    30: [178, 127, 135],
    31: [120, 185, 128],
    32: [146, 111, 194],
    33: [44, 160, 44],
    34: [112, 128, 144],
    35: [96, 207, 209],
    36: [227, 119, 194],
    37: [213, 92, 176],
    38: [94, 106, 211],
    39: [82, 84, 163],
    40: [100, 85, 144]}

SCAN_ID_COLOR = {
    0: [0, 0, 0],
    1: [0, 128, 0],
    2: [128, 0, 0],  # Ideally, Should never be used. Never predict Unknown.
}


def read_tensor_from_file(filename):
    with open(filename, 'rb') as fin:
        depth = struct.unpack('<Q', fin.read(8))[0]
        height = struct.unpack('<Q', fin.read(8))[0]
        width = struct.unpack('<Q', fin.read(8))[0]
        grid: np.ndarray = np.fromfile(fin, dtype=np.uint8, count=-1, sep='')
        grid = grid.reshape((depth, height, width))
        return grid


def make_box_mesh(box_min, box_max, color):
    vertices = [
        np.array([box_max[0], box_max[1], box_max[2], color[0], color[1], color[2]]),
        np.array([box_min[0], box_max[1], box_max[2], color[0], color[1], color[2]]),
        np.array([box_min[0], box_min[1], box_max[2], color[0], color[1], color[2]]),
        np.array([box_max[0], box_min[1], box_max[2], color[0], color[1], color[2]]),
        np.array([box_max[0], box_max[1], box_min[2], color[0], color[1], color[2]]),
        np.array([box_min[0], box_max[1], box_min[2], color[0], color[1], color[2]]),
        np.array([box_min[0], box_min[1], box_min[2], color[0], color[1], color[2]]),
        np.array([box_max[0], box_min[1], box_min[2], color[0], color[1], color[2]])
    ]
    indices = [
        np.array([1, 2, 3], dtype=np.uint32),
        np.array([1, 3, 0], dtype=np.uint32),
        np.array([0, 3, 7], dtype=np.uint32),
        np.array([0, 7, 4], dtype=np.uint32),
        np.array([3, 2, 6], dtype=np.uint32),
        np.array([3, 6, 7], dtype=np.uint32),
        np.array([1, 6, 2], dtype=np.uint32),
        np.array([1, 5, 6], dtype=np.uint32),
        np.array([0, 5, 1], dtype=np.uint32),
        np.array([0, 4, 5], dtype=np.uint32),
        np.array([6, 5, 4], dtype=np.uint32),
        np.array([6, 4, 7], dtype=np.uint32)
    ]
    return vertices, indices


def save_mesh_obj(verts, indices, output_file):
    with open(output_file, 'w') as f:
        for v in verts:
            f.write('v %f %f %f %f %f %f %f\n' % (v[0], v[1], v[2], v[3], v[4], v[5], 0.5))
        f.write('g foo\n')
        for ind in indices:
            f.write('f %d %d %d\n' % (ind[0] + 1, ind[1] + 1, ind[2] + 1))
        f.write('g\n')


def get_bbox_verts(bbox_min, bbox_max):
    verts = [
        np.array([bbox_min[0], bbox_min[1], bbox_min[2]]),
        np.array([bbox_max[0], bbox_min[1], bbox_min[2]]),
        np.array([bbox_max[0], bbox_max[1], bbox_min[2]]),
        np.array([bbox_min[0], bbox_max[1], bbox_min[2]]),

        np.array([bbox_min[0], bbox_min[1], bbox_max[2]]),
        np.array([bbox_max[0], bbox_min[1], bbox_max[2]]),
        np.array([bbox_max[0], bbox_max[1], bbox_max[2]]),
        np.array([bbox_min[0], bbox_max[1], bbox_max[2]])
    ]
    return verts


def rotation(axis, angle):
    rot = np.eye(4)
    c = np.cos(-angle)
    s = np.sin(-angle)
    t = 1.0 - c
    axis /= compute_length_vec3(axis)
    x = axis[0]
    y = axis[1]
    z = axis[2]
    rot[0, 0] = 1 + t * (x * x - 1)
    rot[0, 1] = z * s + t * x * y
    rot[0, 2] = -y * s + t * x * z
    rot[1, 0] = -z * s + t * x * y
    rot[1, 1] = 1 + t * (y * y - 1)
    rot[1, 2] = x * s + t * y * z
    rot[2, 0] = y * s + t * x * z
    rot[2, 1] = -x * s + t * y * z
    rot[2, 2] = 1 + t * (z * z - 1)
    return rot


def compute_length_vec3(vec3):
    return math.sqrt(vec3[0] * vec3[0] + vec3[1] * vec3[1] + vec3[2] * vec3[2])


def create_cylinder_mesh(radius, p0, p1, stacks=10, slices=10):
    verts = []
    indices = []
    diff = (p1 - p0).astype(np.float32)
    height = compute_length_vec3(diff)
    for i in range(stacks + 1):
        for i2 in range(slices):
            theta = i2 * 2.0 * math.pi / slices
            pos = np.array([radius * math.cos(theta), radius * math.sin(theta), height * i / stacks])
            verts.append(pos)
    for i in range(stacks):
        for i2 in range(slices):
            i2p1 = math.fmod(i2 + 1, slices)
            indices.append(np.array([(i + 1) * slices + i2, i * slices + i2, i * slices + i2p1], dtype=np.uint32))
            indices.append(
                np.array([(i + 1) * slices + i2, i * slices + i2p1, (i + 1) * slices + i2p1], dtype=np.uint32))
    transform = np.eye(4)
    va = np.array([0, 0, 1], dtype=np.float32)
    vb = diff
    vb /= compute_length_vec3(vb)
    axis = np.cross(vb, va)
    angle = np.arccos(np.clip(np.dot(va, vb), -1, 1))
    if angle != 0:
        if compute_length_vec3(axis) == 0:
            dotx = va[0]
            if (math.fabs(dotx) != 1.0):
                axis = np.array([1, 0, 0]) - dotx * va
            else:
                axis = np.array([0, 1, 0]) - va[1] * va
            axis /= compute_length_vec3(axis)
        transform = rotation(axis, -angle)
    transform[:3, 3] += p0
    verts = [np.dot(transform, np.array([v[0], v[1], v[2], 1.0])) for v in verts]
    verts = [np.array([v[0], v[1], v[2]]) / v[3] for v in verts]

    return verts, indices


def voxel2obj(voxel_grid, output_file, is_semantic_else_scan):
    """
    voxel_grid: numpy array (x,y,z), in which instance/label id
    output_file: string
    is_semantic_else_scan: change palette to use
    """
    scale = 1
    offset = [0, 0, 0]
    verts = []
    indices = []
    if is_semantic_else_scan:
        colors = ID_COLOR  # NYU40
    else:
        colors = SCAN_ID_COLOR
    for z in range(voxel_grid.shape[2]):
        for y in range(voxel_grid.shape[1]):
            for x in range(voxel_grid.shape[0]):
                if voxel_grid[x, y, z] > 0:
                    box_min = (np.array([x, y, z]) - 0.05) * scale + offset
                    box_max = (np.array([x, y, z]) + 0.95) * scale + offset
                    box_verts, box_ind = make_box_mesh(box_min, box_max,
                                                       np.array(colors[int(voxel_grid[x, y, z] % 41)]))
                    if voxel_grid[x, y, z] > 41:
                        print("Value higher than 41")
                    cur_num_verts = len(verts)
                    box_ind = [x + cur_num_verts for x in box_ind]
                    verts.extend(box_verts)
                    indices.extend(box_ind)
    save_mesh_obj(verts, indices, output_file)


if __name__ == '__main__':
    input_dir = opt.input_path
    output_dir = opt.output_path
    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)
    all_bin_files = [f for f in listdir(input_dir) if isfile(join(input_dir, f)) and f.endswith('.bin')]
    print('Files to process: ', len(all_bin_files))
    for input_file in all_bin_files:
        print('processing %s' % input_file)
        # Treat all non scan files as semantic. Dont check for 'semantic', for backward compatibility with Angela work.
        is_semantic_else_scan = 'scan' not in input_file
        labelled_grid_bin = read_tensor_from_file(join(input_dir, input_file))
        output_file = input_file.replace(".bin", ".obj")
        output_file = join(output_dir, output_file)
        voxel2obj(labelled_grid_bin, output_file, is_semantic_else_scan)
