from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir = '.'
dataType = 'dsb2018'
annFile='{}/annotations/instances_{}_train.json'.format(dataDir,dataType)

coco = COCO(annFile)

cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

catIds = coco.getCatIds(catNms=['nuclei']);
imgIds = coco.getImgIds(catIds=catIds);

for i in range(1, 536):
    imgIds = coco.getImgIds(imgIds=[i])
    img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)

    for ann in anns:
        print(min(ann["segmentation"][0][::2]))

