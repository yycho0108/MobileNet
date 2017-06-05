from pycocotools.coco import COCO
import numpy as np
import cv2
import os

class COCOLoader(object):
    def __init__(self, coco_root, coco_type):
        self.coco_root = coco_root
        self.coco_type = coco_type
        self.ann_dir = os.path.join(coco_root, 'annotations', 'instances_%s.json' % coco_type)
        self.coco = COCO(self.ann_dir)
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        #catAll = coco.getCatIds(catNms=names)
        self.all_cats = {cat['id'] : cat['name'] for cat in self.cats}
        #print self.all_cats
        #print len(self.all_cats)

        self.voc_cats = ['airplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','dining table',
                'dog','horse','motorcycle','person','potted plant', 'sheep', 'couch', 'train', 'tv']

        self.catVOC = self.coco.getCatIds(catNms=self.voc_cats)
    def list_image_sets(self):
        return self.voc_cats
    def list_all(self):
        ids = None
        for i, catId in enumerate(self.catVOC):
            if i==0:
                ids = set(self.coco.catToImgs[catId])
            else:
                ids |= set(self.coco.catToImgs[catId])
        #print 'voc-ids length', len(ids)
        return list(ids)
        #return self.coco.getImgIds()
    def grab(self, id):
        img = self.coco.imgs[id]
        img_path = os.path.join(self.coco_root, self.coco_type, img['file_name'])
        annIds = self.coco.getAnnIds(imgIds = id, catIds=self.catVOC, iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        boxs = []
        lbls = []

        W,H = int(img['width']), int(img['height'])

        for obj in anns:
            box = obj['bbox']
            cat = obj['category_id'] # I'm fairly convinced that it's one-indexed

            cat_name = self.all_cats[cat]
            if not cat_name in self.voc_cats:
                continue

            obj_id = obj['id']
            x,y,w,h = map(lambda x : float(x), box)
            box = [y/H, x/W, (y+h)/H, (x+w)/W]
            lbl = self.voc_cats.index(cat_name)
            boxs.append(box)
            lbls.append(lbl)

        #print self.all_cats
        #print id
        #print [obj['category_id'] for obj in anns]
        #print [self.all_cats[obj['category_id']] for obj in anns]
        #msg = img_path + ' ; ' + (', '.join([str(obj['category_id']) for obj in anns]))
        #assert len(boxs)>0 and len(lbls)>0, msg

        return img_path, np.asarray(boxs, dtype=np.float32), np.asarray(lbls, dtype=np.int32)

if __name__ == "__main__":

    coco_root = os.getenv('COCO_ROOT')
    coco_type = 'train2014'
    coco = COCOLoader(coco_root, coco_type)
    print coco.list_image_sets()
    l = coco.list_all()
    #for im in l:
    #    annIds = coco.coco.getAnnIds(ImgIds = im
    #    coco.coco.loadAnns(annIds)
    img_id = l[0]
    print img_id, type(img_id)
    img_path, boxs, lbls = coco.grab_pair(img_id)
    frame = cv2.imread(img_path)
    print lbls
    h,w = frame.shape[:2]
    for box in boxs:
        y1,x1,y2,x2 = [int(b*s) for (b,s) in zip(box, [h,w,h,w])]
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
    cv2.imshow('frame', frame)
    cv2.waitKey(0)

    #ann = os.path.join(coco_root, 'annotations', 'instances_%s.json' % coco_type)
    #coco = COCO(ann)

    #cats = coco.loadCats(coco.getCatIds())

    ##print cats # probably some dict

    #names = [cat['name'] for cat in cats]

    #print 'category names : ', names

    #catIds = coco.getCatIds(catNms=['person'])
    #catAll = coco.getCatIds(catNms=names)
    #imgIds = coco.getImgIds(catIds = catIds)

    #img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]

    ##print img['file_name']

    #annIds = coco.getAnnIds(imgIds = img['id'], catIds=catAll, iscrowd=None)
    #anns = coco.loadAnns(annIds)
    #frame = cv2.imread(os.path.join(coco_root, coco_type, img['file_name']))

    #H,W = frame.shape[:2]

    #print len(coco.imgs)
    #for img in coco.imgs.itervalues():
    #    frame = cv2.imread(os.path.join(coco_root, coco_type, img['file_name']))
    #    annIds = coco.getAnnIds(imgIds = img['id'], catIds=catAll, iscrowd=None)
    #    anns = coco.loadAnns(annIds)
    #    for obj in anns:
    #        box = obj['bbox']
    #        cat = obj['category_id']
    #        obj_id = obj['id']
    #        x,y,w,h = map(lambda x : float(x), box)

    #        box = [y/H, x/W, (y+h)/H, (x+w)/W]
    #        lbl = int(cat)

    #        print 'box', box
    #        print 'lbl', lbl
    #    break

    ##print coco.imgs.keys()[0]
    ##print coco.imgs.values()[0]
    ##for obj in anns:
    ##    box = obj['bbox']
    ##    cat = obj['category_id']
    ##    obj_id = obj['id']
    ##    x,y,w,h = map(lambda x : float(x), box)

    ##    box = [y/H, x/W, (y+h)/H, (x+w)/W]
    ##    lbl = int(cat)

    ##    print 'box', box
    ##    print 'lbl', lbl

    ##    #x,y,w,h = map(lambda x : (x), [x,y,w,h])
    ##    #cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

    #cv2.imshow('frame', frame)
    #cv2.waitKey(0)
