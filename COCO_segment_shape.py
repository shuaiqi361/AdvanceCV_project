from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17'
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
coco = COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['person', 'dog', 'skateboard']);
imgIds = coco.getImgIds(catIds=catIds);
imgIds = coco.getImgIds(imgIds=[324158])
img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]

# I = io.imread(img['coco_url'])
I = io.imread('%s/images/%s/%s' % (dataDir, dataType, img['file_name']))
plt.axis('off')
plt.imshow(I)
plt.show()
