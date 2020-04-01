from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import cv2
import copy
import math


rescale_size = 300
n_vertices = 128

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17'
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
coco = COCO(annFile)
out_npy_file = '{}/shape_{}_{}.npy'.format(dataDir, dataType, n_vertices)
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
catIds = coco.getCatIds(catNms=nms)
imgIds = coco.getImgIds(catIds=catIds)
annIds = coco.getAnnIds(catIds=catIds)

all_anns = coco.loadAnns(ids=annIds)


def calculateCurvatureThreshold(min_angle=5.):
    min_rad = math.pi / 180. * min_angle
    x = math.cos(math.pi / 2. - min_rad)
    c = math.sin(math.pi / 2. - min_rad)
    y = c / math.tan(2 * min_rad)

    return 1. / (x + y)


def computeCurvatureThreePoints(point1, point2, point3):  # note that the curvature in calculated at point2, the order matters
    a = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    b = np.sqrt((point1[0] - point3[0]) ** 2 + (point1[1] - point3[1]) ** 2)
    c = np.sqrt((point2[0] - point3[0]) ** 2 + (point2[1] - point3[1]) ** 2)
    if a < 1e-6 or b < 1e-6 or c < 1e-6:
        return 0., a + c
    s = (a + b + c) / 2.
    if (s - a) * (s - b) * (s - c) < 0.:
        return 0., a + c
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))

    return 4 * area / (a * b * c), a + c


def normalizeShapeRepresentation(polygons_input, n_vertices, threshold=calculateCurvatureThreshold()):
    polygons = copy.deepcopy(polygons_input)
    total_vertices = len(polygons) // 2
    curvature_thres = threshold
    if total_vertices == n_vertices:
        # print('direct return')
        return polygons
    elif n_vertices * 0.25 <= total_vertices < n_vertices:
        while(len(polygons) < n_vertices * 2):
            max_idx = -1
            max_dist = 0.
            insert_coord = [-1, -1]
            for i in range(len(polygons) // 2):
                x1 = polygons[2 * i]
                y1 = polygons[2 * i + 1]
                x2 = polygons[(2 * i + 2) % len(polygons)]
                y2 = polygons[(2 * i + 3) % len(polygons)]
                dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if dist > max_dist:
                    max_idx = (2 * i + 2) % len(polygons)
                    max_dist = dist
                    insert_coord[0] = (x1 + x2) / 2
                    insert_coord[1] = (y1 + y2) / 2

            polygons.insert(max_idx, insert_coord[1])
            polygons.insert(max_idx, insert_coord[0])

        # print('less than: ', n_vertices)
        return polygons

    elif n_vertices < total_vertices <= n_vertices * 2:
        visited = [0 for i in range(len(polygons))]
        while(len(polygons) > n_vertices * 2):
            min_idx_curv = -1
            min_curv = 0.
            min_idx_side = -1
            min_side = math.inf
            min_side_curv = 100.
            for i in range(len(polygons) // 2):
                if visited[(2 * i + 2) % len(polygons)] == 1:
                    continue
                point1 = (polygons[2 * i], polygons[2 * i + 1])
                point2 = (polygons[(2 * i + 2) % len(polygons)], polygons[(2 * i + 3) % len(polygons)])
                point3 = (polygons[(2 * i + 4) % len(polygons)], polygons[(2 * i + 5) % len(polygons)])
                curvature, side = computeCurvatureThreePoints(point1, point2, point3)
                if side < min_side and curvature < curvature_thres:
                    min_idx_side = (2 * i + 2) % len(polygons)
                    min_side = side

                elif side < min_side and curvature >= curvature_thres:
                    min_idx_side = (2 * i + 2) % len(polygons)
                    visited[min_idx_side] = 1
                    visited[(2 * i + 3) % len(polygons)] = 1
                # if curvature < min_curv:
                #     min_idx_curv = (2 * i + 2) % len(polygons)
                #     min_curv = curvature

            del polygons[min_idx_side]
            del polygons[min_idx_side]
            if np.prod(visited) == 1:
                return None
            # if min_side_curv < curvature_thres:
            #     del polygons[min_idx_side]
            #     del polygons[min_idx_side]
            #     del visited[min_idx_side]
            #     del visited[min_idx_side]
            # else:
            #     visited[min_idx_side] = 1
            #     visited[min_idx_side + 1] = 1
            # del polygons[min_idx]
            # del polygons[min_idx]

        # print('more than: ', n_vertices)
        return polygons

    else:
        # print('return none.')
        return None




counter_iscrowd = 0
counter_total = 0
counter_poor = 0
length_polygons = []
curvature_thres = calculateCurvatureThreshold(min_angle=2.5)
COCO_shape_matrix = np.zeros(shape=(n_vertices * 2, 0))
for annotation in all_anns:
    if annotation['iscrowd'] == 1:
        counter_iscrowd += 1
        continue

    img = coco.loadImgs(annotation['image_id'])[0]
    image_name = '%s/images/%s/%s' % (dataDir, dataType, img['file_name'])
    w_img = img['width']
    h_img = img['height']

    polygons = annotation['segmentation'][0]
    bbox = annotation['bbox']  #top-left corner coordinates, width and height convention

    shape_list = normalizeShapeRepresentation(polygons, n_vertices, threshold=curvature_thres)
    if shape_list is None:
        counter_poor += 1
        continue
    # print('original list size: ', len(polygons))
    # print('returned list size: ', len(shape_list))
    assert len(shape_list) == n_vertices * 2
    counter_total += 1

    # image = cv2.imread(image_name)
    # bound_image = image[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
    # bound_image = cv2.resize(bound_image, dsize=(rescale_size, rescale_size))
    # bound_ref = cv2.resize(bound_image, dsize=(rescale_size, rescale_size))

    norm_shape = shape_list

    for j in range(n_vertices):
        # norm_shape[2 * j] = max(shape_list[2 * j] - bbox[0], 0.) / bbox[2] * rescale_size * 1.
        # norm_shape[2 * j + 1] = max(shape_list[2 * j + 1] - bbox[1], 0.) / bbox[3] * rescale_size * 1.
        norm_shape[2 * j] = max(shape_list[2 * j] - bbox[0], 0.) / bbox[2] * 1.
        norm_shape[2 * j + 1] = max(shape_list[2 * j + 1] - bbox[1], 0.) / bbox[3] * 1.


        # x = int(norm_shape[2 * j])
        # y = int(norm_shape[2 * j + 1])
        # cv2.circle(bound_image, center=(x, y), radius=2, color=(0, 0, 255), thickness=2)
        # cv2.putText(bound_image, text=str(j + 1), org=(x, y), color=(0, 255, 255),
        #             fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=0.8, thickness=1)

    # norm_polygon = polygons
    # for j in range(len(polygons) // 2):
    #     norm_polygon[2 * j] = max(polygons[2 * j] - bbox[0], 0.) / bbox[2] * rescale_size * 1.
    #     norm_polygon[2 * j + 1] = max(polygons[2 * j + 1] - bbox[1], 0.) / bbox[3] * rescale_size * 1.
    #
    #     x = int(norm_polygon[2 * j])
    #     y = int(norm_polygon[2 * j + 1])
    #     cv2.circle(bound_ref, center=(x, y), radius=2, color=(0, 0, 255), thickness=2)
    #     cv2.putText(bound_ref, text=str(j + 1), org=(x, y), color=(0, 255, 255),
    #                 fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=0.8, thickness=1)

    # concat_image = np.zeros((rescale_size, 2 * rescale_size, 3), dtype=np.uint8)
    # concat_image[:, 0:rescale_size, :] = bound_ref
    # concat_image[:, rescale_size:, :] = bound_image
    # cv2.imshow('Compare Image', concat_image)
    # cv2.waitKey()

    # construct data matrix
    # print(norm_shape)
    atom = np.expand_dims(np.array(norm_shape), axis=1)
    # print(atom.shape)
    # print(COCO_shape_matrix.shape)
    COCO_shape_matrix = np.concatenate((COCO_shape_matrix, atom), axis=1)


print('Total valid shape: ', counter_total)
print('Poor shape: ', counter_poor)
print('Iscrowd: ', counter_iscrowd)
print('Total number: ', counter_poor + counter_iscrowd + counter_total)
print('Size of shape matrix: ', COCO_shape_matrix.shape)
np.save(out_npy_file, COCO_shape_matrix)