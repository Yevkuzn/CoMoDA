# Author: Yevhen Kuznietsov
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)


from __future__ import absolute_import, division, print_function


import os
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import transforms


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            if path.find("yolo") == -1:
                return img.convert('RGB')
            else:
                return img.convert("L")


class SegVelDataset(data.Dataset):
    """ The dataset class for loading data required for adaptation
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 seg_suffix="yolov5m3",
                 img_ext='.jpg',
                 load_mask=True):

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.frame_idxs = frame_idxs
        self.img_ext = img_ext
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        self.load_mask = load_mask
        self.seg_suffix = seg_suffix

        self.resize = {}
        # resize the possibly moving objects masks via nearest neighbour interpolation
        self.resize["seg"] = transforms.Resize((self.height, self.width), interpolation=Image.NEAREST)
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s), interpolation=Image.ANTIALIAS)

        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

        # intrinsics
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)


    def preprocess(self, inputs):
        """Resize colour images and the semantic masks to the required scales
        No augmentation is performed for adaptation
        """
        for k in list(inputs):
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)

        if self.load_mask:
            for i in self.frame_idxs:
                # possibly dynamic objects are masked out only at top scale
                inputs[("seg", i, 0)] = self.resize["seg"](inputs[("seg", i, -1)])
                inputs[("seg", i, 0)] = self.to_tensor(inputs[("seg", i, 0)])


    def __len__(self):
        return len(self.filenames)


    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics.
            ("trans:<frame_id>-<frame_id>")         for camera translation between two frames
            ("seg", <frame_id>, <scale>)            for mask to ignore possibly moving objects

        <frame_id> is an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index'

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        line = self.filenames[index].split()
        folder = line[0]
        frame_index = int(line[1])
        side = line[2]

        # load velocity data
        kinetics_vec = torch.FloatTensor(self.get_kinetics(folder, frame_index, side))
        inputs[("trans:0-1")] = kinetics_vec[0] * kinetics_vec[1]
        inputs[("trans:1-2")] = kinetics_vec[2] * kinetics_vec[3]

        # load depth and mask triplets, seems to be faster than loading images one by one
        triplet = self.get_triplet(folder, frame_index, side)
        if self.load_mask:
            seg_triplet = self.get_seg_triplet(folder, frame_index, side)

        # split triplet images
        for i in self.frame_idxs:
            inputs[("color", i, -1)] = Image.fromarray(triplet[:, :, (i+1)*3: (i+2)*3], "RGB")
            if self.load_mask:
                inputs[("seg", i, -1)] = Image.fromarray(seg_triplet[:, :, i+1], "L")

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)
            inv_K = np.linalg.pinv(K)
            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        self.preprocess(inputs)

        # free some memory..
        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            if self.load_mask:
                del inputs[("seg", i, -1)]

        return inputs


    def get_triplet(self, folder, frame_index, side):
        triplet = self.loader(self.get_triplet_path(folder, frame_index, side))
        triplet_array = np.array(triplet)
        triplet = np.split(triplet_array, 3, axis=1)
        triplet = np.concatenate(triplet, axis=2)
        return triplet


    def get_triplet_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        folder = folder.split("/")[1] + "_0" + str(self.side_map[side])
        triplet_path = os.path.join(self.data_path, folder, f_str)
        return triplet_path


    def get_kinetics(self, folder, frame_index, side):
        path = self.get_kin_path(folder, frame_index, side)
        f = open(path, "r")
        kinetics_vec = f.readline().split(" ")
        kinetics_vec = [float(v) for v in kinetics_vec]
        return kinetics_vec


    def get_kin_path(self, folder, frame_index, side):
        f_str = "{:010d}_kin.txt".format(frame_index)
        folder = folder.split("/")[1] + "_0" + str(self.side_map[side])
        kin_path = os.path.join(self.data_path, folder, f_str)
        return kin_path


    def get_seg_triplet(self, folder, frame_index, side):
        triplet = self.loader(self.get_seg_path(folder, frame_index, side))
        triplet_array = np.array(triplet)
        triplet = np.split(triplet_array, 3, axis=1)
        triplet = np.stack(triplet, axis=2)
        return triplet


    def get_seg_path(self, folder, frame_index, side):
        f_str = "{:010d}-{}.png".format(frame_index, self.seg_suffix)
        folder = folder.split("/")[1] + "_0" + str(self.side_map[side])
        seg_path = os.path.join(self.data_path, folder, f_str)
        return seg_path
