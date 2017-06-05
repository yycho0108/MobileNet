from pycocotools.coco import COCO
import numpy as np
import cv2
import os

if __name__ == "__main__":

    coco_root = os.getenv('COCO_ROOT')
    coco_type = 'train2014'
    ann = os.path.join(coco_root, 'annotations', 'instances_%s.json' % coco_type)
    coco = COCO(ann)

    cats = coco.loadCats(coco.getCatIds())

    #print cats # probably some dict

    names = [cat['name'] for cat in cats]

    #print 'category names : ', names

    catIds = coco.getCatIds(catNms=['person'])
    catAll = coco.getCatIds(catNms=names)
    imgIds = coco.getImgIds(catIds = catIds)

    img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]

    #print img['file_name']

    annIds = coco.getAnnIds(imgIds = img['id'], catIds=catAll, iscrowd=None)
    anns = coco.loadAnns(annIds)
    frame = cv2.imread(os.path.join(coco_root, coco_type, img['file_name']))

    H,W = frame.shape[:2]

    print len(coco.imgs)
    for img in coco.imgs.itervalues():
        frame = cv2.imread(os.path.join(coco_root, coco_type, img['file_name']))
        annIds = coco.getAnnIds(imgIds = img['id'], catIds=catAll, iscrowd=None)
        anns = coco.loadAnns(annIds)
        for obj in anns:
            box = obj['bbox']
            cat = obj['category_id']
            obj_id = obj['id']
            x,y,w,h = map(lambda x : float(x), box)

            box = [y/H, x/W, (y+h)/H, (x+w)/W]
            lbl = int(cat)

            print 'box', box
            print 'lbl', lbl
        break

    #print coco.imgs.keys()[0]
    #print coco.imgs.values()[0]
    #for obj in anns:
    #    box = obj['bbox']
    #    cat = obj['category_id']
    #    obj_id = obj['id']
    #    x,y,w,h = map(lambda x : float(x), box)

    #    box = [y/H, x/W, (y+h)/H, (x+w)/W]
    #    lbl = int(cat)

    #    print 'box', box
    #    print 'lbl', lbl

    #    #x,y,w,h = map(lambda x : (x), [x,y,w,h])
    #    #cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.imshow('frame', frame)
    cv2.waitKey(0)
