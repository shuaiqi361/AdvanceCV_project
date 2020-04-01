import matplotlib.pyplot as plt
import numpy as np
import cv2
from random import randint


n_atom = 64
n_vertices = 32
n_atom_row = 8
n_atom_col = 8
basis_size = 100
padding = 10
alpha = 0.02

if n_vertices == 32:
    shape_data = np.load('/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17/shape_val2017_32.npy')
elif n_vertices == 64:
    shape_data = np.load('/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17/shape_val2017_64.npy')
elif n_vertices == 16:
    shape_data = np.load('/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17/shape_val2017_16.npy')
else:
    print('Not implemented, try n_vertices: 16, 32, 64')
    exit()
out_dict = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17/shape_codes/sparsity/dict_val2017_v{}_b{}_alpha{}.npy'.format(n_vertices, n_atom, alpha)
out_code = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17/shape_codes/sparsity/code_val2017_v{}_b{}_alpha{}.npy'.format(n_vertices, n_atom, alpha)

learned_dict = np.load(out_dict)
learned_codes = np.load(out_code)
print('Dict shape: ', learned_dict.shape)

canvas = np.ones(shape=(n_atom_row * basis_size, n_atom_col * basis_size, 3), dtype=np.uint8) * 255

basis_idx = 0
for i in range(n_atom_row):
    for j in range(n_atom_col):
        patch = np.ones(shape=(basis_size, basis_size, 3), dtype=np.uint8) * 255
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        contour = np.reshape(learned_dict[basis_idx, :], newshape=(n_vertices, 2))
        contour -= np.min(contour, axis=0, keepdims=True)
        norm_contour = contour / np.max(contour, axis=0, keepdims=True)
        # print(np.min(norm_contour), np.max(norm_contour))
        norm_contour = ((norm_contour * (basis_size - 2. * padding)) + padding).astype(np.int32)

        # print(norm_contour)
        cv2.rectangle(patch, (padding // 2, padding // 2), (basis_size - padding // 2, basis_size - padding // 2), color=(0, 0, 0))
        cv2.polylines(patch, [norm_contour], isClosed=True, color=color, thickness=2)
        canvas[i * basis_size:(i + 1) * basis_size, j * basis_size:(j + 1) * basis_size, :] = patch
        # cv2.imshow('Shapes', patch)
        # cv2.waitKey(0)

        basis_idx += 1
        if basis_idx >= n_atom:
            break
    if basis_idx >= n_atom:
        break

cv2.imshow('Shapes', canvas)
cv2.waitKey(0)
