from utils import *
import os
import sys


voc07_path = '/home/keyi/Documents/Data/VOC_2007/VOC_2007_merge/VOC2007'
voc12_path = '/home/keyi/Documents/Data/VOC_2012/VOCdevkit/VOC2012'
output_folder = '/home/keyi/Documents/courses/AdvancedCV/project/advanced_cv_project'

create_data_lists(voc07_path, voc12_path, output_folder)