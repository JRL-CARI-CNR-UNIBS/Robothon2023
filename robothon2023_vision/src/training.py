#!/usr/bin/env python3

from fastai.vision.all import *
from fastai.vision.core import *
from pathlib import Path
from PIL import Image

import rospy
import numpy as np
from utils import *
import cv2

# def sep_points(coords:array):
#   "Seperate a set of points to groups"
#   kpts = []
#   for i in range(1, int(coords[0]*2), 2):
#     kpts.append([coords[i], coords[i+1]])
#   return tensor(kpts)
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from utils import *
import cv2
import matplotlib.pyplot as plt

from torch import *
import ruamel.yaml

# from vision.core import TensorPoint, PILImage


@dataclass
class ImagesKeypointManager:
    known_keypoints: List[str]
    image_labels: Dict = field(default=dict, init=False)

    def __post_init__(self):
        self.image_labels = {}

    def add_image_label(self, img_name, keypoints):
        if img_name not in self.image_labels:
            self.image_labels[img_name] = []
            for keypoint in self.known_keypoints:
                self.image_labels[img_name].append([keypoints[keypoint]["x"], keypoints[keypoint]["y"]])
                # self.image_labels[img_name].append(keypoints[keypoint]["y"])

    def get_image_label(self, img_name):
        return self.image_labels[img_name]

    def get_y(self, img_name):
        return torch.Tensor(self.image_labels[img_name.split("/")[-1]])

        # return tpnts  # torch.Tensor(self.image_labels[img_name])

        # return self.get_image_label_as_torch(img_name)

    def get_image_label_as_torch(self, img_name):
        tfm = Transform(TensorPoint.create)
        tpnts = tfm(self.image_labels[img_name])

        return tpnts  # torch.Tensor(self.image_labels[img_name])

    def get_all_images_labels(self):
        return self.image_labels

    def get_knwon_keypoints(self):
        known_keypoints = []
        for keypoint in self.known_keypoints:
            known_keypoints.append(f"{keypoint}_x")
            known_keypoints.append(f"{keypoint}_y")
        return known_keypoints

    def create_txt(self,folder_path,img_name):
        keyp = self.get_image_label(img_name)

        file = Path(folder_path+img_name.split(".")[0]+".yaml")

        yaml = ruamel.yaml.YAML(typ='rt')
        yaml.preserve_quotes = True
        with open(file, 'wb') as f:
            yaml.dump(keyp, f)
    def create_txt2(self,folder_path,img_name):
        folder_path+="prova/"
        keyp = self.get_image_label(img_name)
        dim = {"red_button":[20,20],
                 "blue_button":[20,20],
                 "red_plug":[30,30],
                 "door_handle":[40,40],
                 "slider":[5,13]}
        file = Path(folder_path+img_name.split(".")[0]+".txt")
        str = ""
        for id,keypoint in enumerate(self.image_labels[img_name]):
            print(keypoint)
            print(id)
            # print(dim[keypoint][1])
            # print(self.image_labels[keypoint][0])
            center_x=keypoint[0]/1280.0
            center_y=keypoint[1]/720.0
            width=dim[self.known_keypoints[id]][0]/1280.0
            height=dim[self.known_keypoints[id]][1]/720.0
            str += f"{id} {center_x} {center_y} {width} {height} \n"
            #
            # yaml = ruamel.yaml.YAML(typ='rt')
            # yaml.preserve_quotes = True
        with open(file, 'w') as f:
            f.write(str)


def get_y(fname):
    # labels_path = rospy.get_param("labels_path")
    global labels_path # = "/home/samuele/projects/robothon2023_ws/src/Robothon2023/robothon2023_vision/labels/"
    yaml = ruamel.yaml.YAML(typ='rt')
    yaml.preserve_quotes = True
    fname = str(fname)
    fname = fname.split("/")[-1]
    fname = labels_path +  fname.replace("png","yaml")
    data = yaml.load(Path(fname))
    return torch.Tensor(data)


def get_ip(img: PILImage, pts: array): return TensorPoint(pts, sz=img.size)


def main():
    rospy.init_node('keypoints_labeling')

    if not check_params_existence(
            ["dataset_path", "fig_prefix_name", "known_keypoints", "~image_label","labels_path"]):
        return 0

    fig_prefix_name = rospy.get_param("fig_prefix_name")
    dataset_path = rospy.get_param("dataset_path")
    # print(dataset_path)
    image_label = rospy.get_param("~image_label")
    known_keyp = rospy.get_param("known_keypoints")
    imgs_keyp = ImagesKeypointManager(known_keyp)
    global labels_path
    labels_path = rospy.get_param("labels_path")
    print(labels_path)
    for image in image_label:
        imgs_keyp.add_image_label(image, image_label[image])
        imgs_keyp.create_txt(labels_path,image)
        imgs_keyp.create_txt2(labels_path,image)



    dataset_folder_path = Path(dataset_path)
    imgs = get_image_files(dataset_folder_path)

    for t in range(0,10):
        print(t)
        # img = PILImage.create(dataset_path + f"frame_{t}.png")
        # ax = img.show(figsize=(12, 12))
        #
        # get_ip(img,get_y(labels_path+f"frame_{t}.png")).show(ctx=ax)
        # plt.show()

    # print(imgs)
    # item_tfms = [Resize(448, method='squish')]
    item_tfms = []
    # batch_tfms = [Flip(), Rotate()]  # , Zoom(), Warp(), ClampBatch()]
    batch_tfms = []

    dblock = DataBlock(blocks=(ImageBlock, PointBlock),
                       get_items=get_image_files,
                       splitter=RandomSplitter(),
                       get_y=get_y,
                       item_tfms=item_tfms,
                       batch_tfms=batch_tfms)
    dls = dblock.dataloaders(dataset_folder_path, bs=2)

    # dls.show_batch(max_n = 6)
    # plt.show()

    # body = create_body(resnet18(), pretrained=True)
    #
    # head = create_head(nf=1024, n_out=len(known_keyp))
    # arch = nn.Sequential(body, head)
    #
    # learn = Learner(dls, arch, loss_func=MSELossFlat(), splitter=_resnet_split,
    #                 opt_func=ranger)
    # learn.freeze()
    # learn.lr_find()

    learn = vision_learner(dls, resnet18, loss_func=MSELossFlat())
    learn.summary()

    learn.freeze()
    learn.fit_flat_cos(50, 1e-7)
    learn.unfreeze()
    learn.lr_find()
    learn.fit_flat_cos(10, 1e-7)

    # learn.model()

    # learn.remove_cb(ProgressCallback)

    # learn.fine_tune(100,learn.lr_find(),freeze_epochs=100)
    learn.show_results()
    learn.save('model')


    plt.show()


    # dblock.summary(dataset_folder_path)
    # # print()
    # imgs = get_image_files(dataset_folder_path)
    # print(imgs)
    # img.show()
    # cv2.waitKey(-1)
    # tp = get_ip(img, ip)
    # pnt_img = TensorImage(im.resize((28, 35)))
    #
    #     y = get_keyp(name)
    #     for x in y:
    #         if x[0] < im.size[0]:
    #             if x[0] < 0:
    #                 bad_imgs.append(name)
    #             if x[1] < im.size[1]:
    #                 if x[1] < 0:
    #                     bad_imgs.append(name)
    #             else:
    #                 bad_imgs.append(name)
    #         else:
    #             bad_imgs.append(name)
    # item_tfms = [Resize(448, method='squish')]
    # batch_tfms = [Flip(), Rotate(), Zoom(), Warp(), ClampBatch()]
    # dblock = DataBlock(blocks=(ImageBlock, PointBlock),
    #                    get_items=get_image_files,
    #                    splitter=RandomSplitter(),
    #                    get_y=lambda x: get_y(labels_path, x),
    #                    item_tfms=item_tfms,
    #                    batch_tfms=batch_tfms)
    # dblock.summary('')
    # dls = dblock.dataloaders('', path='', bs=bs)
    # dls.show_batch(max_n=8, figsize=(12, 12))
    # body = create_body(resnet18, pretrained=True)
    # head = create_head(nf=1024, n_out=18);
    # arch = nn.Sequential(body, head)
    # print(arch[1:])
    # apply_init(arch[1], nn.init.kaiming_normal_)
    # learn = Learner(dls, arch, loss_func=MSELossFlat(), splitter=_resnet_split,
    #                 opt_func=ranger)
    # learn.freeze()
    # learn.lr_find()
    # print(learn.fit_flat_cos(5, 1e-2))
    # learn.show_results()


if __name__ == "__main__":
    main()
