import pandas as pd
import os
from bs4 import BeautifulSoup
from more_itertools import unique_everseen
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import io


root_dir = os.environ["VOC_ROOT"]
img_dir = os.path.join(root_dir, 'JPEGImages/')
ann_dir = os.path.join(root_dir, 'Annotations')
set_dir = os.path.join(root_dir, 'ImageSets', 'Main')

class VOCLoader(object):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'JPEGImages/')
        self.ann_dir = os.path.join(root_dir, 'Annotations/')
        self.set_dir = os.path.join(root_dir, 'ImageSets', 'Main')

    def list_image_sets(self):
        return [
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train',
            'tvmonitor']
    def list_types(self):
        return ['train', 'val', 'trainval', 'test']

    def imgs_from_category(self, cat_name, dataset, as_list=False):
        filename = os.path.join(self.set_dir, cat_name + "_" + dataset + ".txt")
        df = pd.read_csv(
            filename,
            delim_whitespace=True,
            header=None,
            names=['filename', 'true'])
        if(as_list):
            df = df[df['true'] == 1]
            return df['filename'].values
        return df

    def annotation_file_from_img(self, img_name):
        return os.path.join(self.ann_dir, img_name) + '.xml'

    def img_from_annotation(self, annot):
        img_file = annot.findChild('filename').contents[0]
        return os.path.join(self.img_dir, img_file)

    def grab_pair(self, basename):
        img_name = os.path.join(self.img_dir, basename + '.jpg')
        xml = ""
        with open(os.path.join(self.ann_dir, basename + '.xml')) as f:
            xml = f.readlines()
        xml = ''.join([line.strip('\t') for line in xml])
        ann = BeautifulSoup(xml)
        return img_name, ann

    def load_annotation(self, img_filename):
        """
        Load annotation file for a given image.

        Args:
            img_name (string): string of the image name, relative to
                the image directory.

        Returns:
            BeautifulSoup structure: the annotation labels loaded as a
                BeautifulSoup data structure
        """
        xml = ""
        with open(self.annotation_file_from_img(img_filename)) as f:
            xml = f.readlines()
        xml = ''.join([line.strip('\t') for line in xml])
        return BeautifulSoup(xml)

    def load_img(self, img_filename, path_only=False):
        """
        Load image from the filename. Default is to load in color if
        possible.

        Args:
            img_name (string): string of the image name, relative to
                the image directory.

        Returns:
            np array of float32: an image as a numpy array of float32
        """
        img_filename = os.path.join(self.img_dir, img_filename + '.jpg')
        if path_only:
            return img_filename
        img = skimage.img_as_float(io.imread(
            img_filename)).astype(np.float32)
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        return img
    def list_all(self):
        ## CUSTOM FUNCTION -- DO NOT USE
        with open(os.path.join(self.root_dir, 'list.txt')) as f:
            return [l.strip() for l in f.readlines()]
    def annotations(self):
        for fn in os.listdir(self.ann_dir):
            filepath = os.path.join(self.ann_dir, fn)
            if os.path.isfile(filepath):
                xml = ""
                with open(filepath) as f:
                    xml = f.readlines()
                xml = ''.join([line.strip('\t') for line in xml])
                yield BeautifulSoup(xml)
