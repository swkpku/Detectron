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
imgIds = coco.getImgIds(imgIds=[500])
img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]

I = io.imread('%s/nuclei_train2018/%s' % (dataDir, img['file_name']))
plt.axis('off')
plt.imshow(I)
plt.show()


# load and display instance annotations
plt.imshow(I)
plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)

print(anns)

coco.showAnns(anns)
plt.show()
