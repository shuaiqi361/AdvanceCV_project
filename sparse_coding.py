import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import DictionaryLearning, dict_learning_online

n_atom = 64
n_vertices = 32
n_atom_row = 8
n_atom_col = 8
alpha = 0.002

if n_vertices == 32:
    shape_data = np.load('/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17/shape_val2017_32.npy')
elif n_vertices == 64:
    shape_data = np.load('/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17/shape_val2017_64.npy')
elif n_vertices == 16:
    shape_data = np.load('/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17/shape_val2017_16.npy')
elif n_vertices == 128:
    shape_data = np.load('/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17/shape_val2017_128.npy')
else:
    print('Not implemented, try n_vertices: 16, 32, 64')
    exit()
out_dict = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17/shape_codes/sparsity/dict_val2017_v{}_b{}_alpha{}.npy'.format(n_vertices, n_atom, alpha)
out_code = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17/shape_codes/sparsity/code_val2017_v{}_b{}_alpha{}.npy'.format(n_vertices, n_atom, alpha)
n_dim, n_data = shape_data.shape
print('Shape data dims: ', shape_data.shape)
# dict_learner = DictionaryLearning(n_components=n_atom, alpha=1., max_iter=500)
learned_codes, learned_dict = dict_learning_online(np.transpose(shape_data), n_components=n_atom,
                                    alpha=alpha, n_iter=2000, batch_size=50, return_code=True)

print("Learned dictionary dim: ", learned_dict.shape)
print("Learned codes dim: ", learned_codes.shape)

# calculate the reconstruction error
error = np.sum((np.matmul(learned_codes, learned_dict) - np.transpose(shape_data)) ** 2) / n_data
print('reconstruction error(frobenius): ', error)

np.save(out_dict, learned_dict)
np.save(out_code, learned_codes)