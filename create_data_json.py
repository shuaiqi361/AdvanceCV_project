from utils import *
import os
import sys


voc07_path = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/VOC_2007/VOCdevkit/VOC2007'
voc12_path = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/VOC_2012/VOCdevkit/VOC2012'
output_folder = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data'

create_data_lists(voc07_path, voc12_path, output_folder)