import numpy as np
import cv2
import os

from collections import defaultdict

choice = np.random.choice

def random_swap(a,b,p=0.5):
    if np.random.random()<0.5:
        a,b=b,a
    return a,b

class MkLoader(object):
    def __init__(self, mk_root, split_ratio=1.0):
        self.mk_root = mk_root
        self.im_root = os.path.join(mk_root, 'images')

        self.train = {
                'good' : defaultdict(lambda:[]),
                'junk' : [],
                'dist' : []
                }

        self.test = {
                'good' : defaultdict(lambda:[]),
                'junk' : [],
                'dist' : []
                }

        with open(os.path.join(self.mk_root, 'list.txt'), 'r') as f:
            self.list = [e.strip() for e in f.readlines()]
        np.random.shuffle(self.list)

        n = len(self.list)
        n_train = int(n * split_ratio)
        n_test = n - n_train

        train_list = self.list[:n_train]
        test_list = self.list[n_train:]

        for e in train_list:
            identity = e.split('_')[0]
            if identity == '0000':
                self.train['dist'].append(e)
            elif identity == '-1':
                self.train['junk'].append(e)
            else:
                self.train['good'][identity].append(e)

        for e in test_list:
            identity = e.split('_')[0]
            if identity == '0000':
                self.test['dist'].append(e)
            elif identity == '-1':
                self.test['junk'].append(e)
            else:
                self.test['good'][identity].append(e)

        self.data = {
                'train' : self.train,
                'test' : self.test
                }

    def random_identity(self, type, **kwargs):
        return choice(self.data[type]['good'].keys(), **kwargs)
    def get_same(self, type='train'):
        i = self.random_identity(type)
        good = self.data[type]['good']
        return choice(good[i],size=2, replace=False)
    def get_diff(self, type='train'):
        i_1,i_2 = self.random_identity(type, size=2, replace=False)
        good = self.data[type]['good']
        return choice(good[i_1]), choice(good[i_2])
    def get_dist(self, type='train'):
        i = self.random_identity(type)
        p1, p2 = choice(self.data[type]['good'][i]), choice(self.data[type]['dist'])
        return random_swap(p1,p2)
    def get_junk(self, type='train'):
        i = self.random_identity(type)
        p1, p2 = choice(self.data[type]['good'][i]), choice(self.data[type]['junk'])
        return random_swap(p1,p2)
    def get_img(self, img):
        return cv2.imread(os.path.join(self.im_root,img))[...,::-1]/255.0 #RGB
    def get_batch(self, batch_size, transpose=False, as_img=False, type='train'):
        p = np.random.dirichlet(np.ones(4), size=1).reshape(-1)
        fls = [(self.get_same,1), (self.get_diff,0), (self.get_dist,0), (self.get_junk,0)]
        idx = choice(4,size=batch_size,replace=True,p=p)

        if transpose:
            res = []
        else:
            p1s = []
            p2s = []
            ls = []

        for i in idx:
            f,l = fls[i]

            if as_img:
                p1,p2 = [self.get_img(p) for p in f(type)]
            else:
                p1,p2 = f(type)

            if transpose:
                res.append((p1,p2,l))
            else:
                p1s.append(p1)
                p2s.append(p2)
                ls.append(l)

        if transpose:
            return res
        else:
            return p1s, p2s, ls


def main():

    loader = MkLoader('/home/yoonyoungcho/Downloads/Market-1501-v15.09.15/', split_ratio=1.0)
    #p1s, p2s, ls = loader.get_batch(32)

    #print len(p1s)
    #print len(p2s)
    #print len(ls)

    #for (p1,p2,l) in zip(p1s,p2s,ls):
    #    cv2.imshow('p1', p1)
    #    cv2.imshow('p2', p2)
    #    print 'l', l
    #    cv2.waitKey(0)

    for (p1,p2,l) in loader.get_batch(32, transpose=True, type='train'):
        cv2.imshow('p1', loader.get_img(p1))
        cv2.imshow('p2', loader.get_img(p2))
        print 'l', l
        if cv2.waitKey(0) == 27:
            break

    #sample = {}
    #sample['sm'] = loader.get_same()
    #sample['df'] = loader.get_diff()
    #sample['ds'] = loader.get_dist()
    #sample['jk'] = loader.get_junk()

    #for nm,ims in sample.iteritems():
    #    print nm, ims
    #    frames = [loader.get_img(f) for f in ims]
    #    cv2.imshow('0', frames[0])
    #    cv2.imshow('1', frames[1])
    #    cv2.waitKey(0)

if __name__ == "__main__":
    main()

