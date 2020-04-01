from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pylab
import cv2

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17'
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
coco = COCO(annFile)

cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
# print(nms)
catIds = coco.getCatIds(catNms=nms)
# print('catIds: ', catIds)
imgIds = coco.getImgIds(catIds=catIds)
annIds = coco.getAnnIds(catIds=catIds)

# imgIds = coco.getImgIds(imgIds=[324158])
# img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
all_anns = coco.loadAnns(ids=annIds)
# f_seg = open('/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO_val17_seg.txt', 'w')
counter = 0
length_polygons = []
area_list = []
reso_list = []
width_list = []
height_list = []
for annotation in all_anns:
    if annotation['iscrowd'] == 1:
        continue

    img = coco.loadImgs(annotation['image_id'])[0]
    image_name = '%s/images/%s/%s' % (dataDir, dataType, img['file_name'])
    width = img['width'] * 1.
    height = img['height'] * 1.
    # image = cv2.imread(image_name)
    # I = io.imread(image_name)
    # plt.axis('off')
    # plt.imshow(I)
    # plt.show()
    # new_line = image_name + ' '
    # print(annotation)
    # exit()
    polygons = annotation['segmentation'][0]
    bbox = annotation['bbox']
    # area = annotation['area']  # / width / height
    area = bbox[3] * bbox[2] * 1. / width / height
    # if len(polygons) // 2 > 200:
    #     image = cv2.imread(image_name)
    #     for i in range(len(polygons) // 2):
    #         x = polygons[2 * i]
    #         y = polygons[2 * i + 1]
    #         cv2.circle(image, center=(int(x), int(y)), radius=1, color=(0, 0, 255))
    #         cv2.putText(image, text=str(i), org=(int(x), int(y)), color=(0, 255, 255), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=0.5)
    #     cv2.imshow('image', image)
    #     cv2.waitKey()
    # for vertex in polygons:
    #     new_line += str(vertex) + ' '
    #
    # new_line += '\n'
    # f_seg.write(new_line)
    length_polygons.append(len(polygons) // 2)
    area_list.append(area)
    reso_list.append(width * height)
    width_list.append(width)
    height_list.append(height)
    counter += 1

print(dataType, 'segments: ', counter)
print('mean len: ', np.mean(length_polygons), 'std len: ', np.std(length_polygons))
# print('mean area: ', np.mean(area_list), 'std area: ', np.std(area_list))
print('mean reso: ', np.mean(reso_list), 'std reso: ', np.std(reso_list))
# plt.plot(area_list, length_polygons, 'r+', linewidth=2, markersize=5)
# plt.plot(reso_list, length_polygons, 'r+', linewidth=2, markersize=5)

ax = plt.axes(projection='3d')
ax.plot3D(width_list, height_list, length_polygons, 'ro')
# plt.xlabel('Image width')
# plt.ylabel('Image height')
# plt.zlabel('Number of vertices')
ax.set_xlabel('Image width')
ax.set_ylabel('Image height')
ax.set_zlabel('Number of vertices')
plt.show()