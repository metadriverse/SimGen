import os
import random
import cv2
import numpy as np
import json
from torch.utils.data import Dataset

import sys
sys.path.append('.')
import tqdm

from util import *


class SimpleDataset(Dataset):
    """
    A simple dataset class for loading data from example_data folder
    Supports two modes:
    1. Using arbitrary conditions to generate real_img (img2img mode)
    2. Using text as input to generate conditions (t2i mode)
    """
    def __init__(self,
                 data_root="example_data",
                 mode="img2img",  # "img2img" or "t2i"
                 local_type_list=["depth", "seg"],  # Can be any combination of conditions
                 resolution=512,
                 drop_txt_prob=0.0,
                 keep_all_cond_prob=1.0,
                 drop_all_cond_prob=0.0,
                 drop_each_cond_prob=[0.5, 0.5],  # Should match length of local_type_list
                 ):
        self.data_root = data_root
        self.mode = mode
        self.local_type_list = local_type_list
        self.resolution = resolution
        self.drop_txt_prob = drop_txt_prob
        self.keep_all_cond_prob = keep_all_cond_prob
        self.drop_all_cond_prob = drop_all_cond_prob
        self.drop_each_cond_prob = drop_each_cond_prob

        # Load text labels
        with open(os.path.join(data_root, "text_label.json"), 'r') as f:
            self.text_labels = json.load(f)

        # Get all image filenames
        self.image_files = list(self.text_labels.keys())
        
        # Build data info dictionary
        self.data_infos = []
        for img_file in self.image_files:
            img_dict = {}
            img_dict["text"] = self.text_labels[img_file]
            
            # Set different paths based on mode
            if mode == "img2img":
                # Use conditions to generate real_img
                img_dict["real_img"] = os.path.join(data_root, "real_img", img_file)
                for cond_type in local_type_list:
                    img_dict[cond_type] = os.path.join(data_root, cond_type, img_file)
            else:
                # Use text as input to generate conditions
                for cond_type in local_type_list:
                    img_dict[cond_type] = os.path.join(data_root, cond_type, img_file)
            
            self.data_infos.append(img_dict)

    def keep_and_drop(self, conditions, keep_all_cond_prob, drop_all_cond_prob, drop_each_cond_prob):
        """
        Randomly drop conditions based on probabilities
        Args:
            conditions: list of condition images
            keep_all_cond_prob: probability to keep all conditions
            drop_all_cond_prob: probability to drop all conditions
            drop_each_cond_prob: list of probabilities to drop each condition
        Returns:
            list of kept conditions
        """
        if random.random() < keep_all_cond_prob:
            return conditions
        if random.random() < drop_all_cond_prob:
            return []
        kept_conditions = []
        for i, cond in enumerate(conditions):
            if random.random() > drop_each_cond_prob[i]:
                kept_conditions.append(cond)
        return kept_conditions

    def cond_encoder(self, image):
        """
        Encode conditions into a specific format
        Args:
            image: input condition image
        Returns:
            encoded condition image
        """
        COLOR_DICT_CITY = np.array([
            [0, 0, 0],          # 0: 'ignore'
            [128, 64,128],      # 1: 'road'
            [232, 35, 244],     # 2: 'sidewalk'
            [70, 70, 70],       # 3: 'building'
            [156,102,102],      # 4: 'wall'
            [153,153,190],      # 5: 'fence'
            [153,153,153],      # 6: 'pole'
            [30, 170,250],      # 7: 'traffic light'
            [0,220,  220],      # 8: 'traffic sign'
            [35,142, 107],      # 9: 'vegetation'
            [152,251,152],      # 10: 'terrain'
            [180,130,70],       # 11: 'sky'
            [60, 20, 220],      # 12: 'person'
            [0,  0,  255],      # 13: 'rider'
            [142, 0, 0],        # 14: 'car'
            [70,  0, 0],        # 15: 'truck'
            [100, 60,0],        # 16: 'bus'
            [100, 80,0],        # 17: 'train'
            [230, 0, 0],        # 18: 'motorcycle'
            [32, 11, 119],      # 19: 'bicycle'
            [189, 176, 55],     # 20: 'cross work'
            [255, 255, 255],    # 21: 'lane line'
        ])
        COLOR_DICT_FORE = np.array([
            [0, 0, 0],          # 0: 'ignore'
            [128, 64,128],      # 1: 'road'
            [153,153,190],      # 2: 'fence'
            [60, 20, 220],      # 3: 'person'
            [142, 0, 0],        # 4: 'car'
            [70,  0, 0],        # 5: 'truck'
            [100, 60,0],        # 6: 'bus'
            [230, 0, 0],        # 7: 'motorcycle'
            [32, 11, 119],      # 8: 'bicycle'
            [189, 176, 55],     # 9: 'cross work'
            [255, 255, 255],    # 10: 'lane line'
        ])
        COLOR_DICT_FORE[:,[2,0]] = COLOR_DICT_FORE[:,[0,2]]
        output_cond = np.zeros((image.shape[0], image.shape[1], 3), dtype=int)
        output_cond[:, :, 0] = image[:, :, 0]  # depth
        COLOR_DICT_CITY[:,[2,0]] = COLOR_DICT_CITY[:,[0,2]]
        seg_image = image[:, :, 3:]
        
        colormap = COLOR_DICT_CITY
        colormap_reshaped = colormap.reshape((1, -1, 3))
        distances = np.linalg.norm(seg_image[:, :, np.newaxis, :] - colormap_reshaped, axis=-1)
        index_matrix = np.argmin(distances, axis=-1)

        index_matrix = np.where(index_matrix == 20, 1, index_matrix)
        index_matrix = np.where(index_matrix == 21, 1, index_matrix)

        colormap = COLOR_DICT_FORE
        colormap_reshaped = colormap.reshape((1, -1, 3))
        distances = np.linalg.norm(seg_image[:, :, np.newaxis, :] - colormap_reshaped, axis=-1)
        fore_index_matrix = np.argmin(distances, axis=-1)

        fore_index_matrix = np.where(fore_index_matrix == 9, 1, fore_index_matrix)
        fore_index_matrix = np.where(fore_index_matrix == 10, 1, fore_index_matrix)

        mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
        foreground = [1,5,12,13,14,15,16,17,18,19,20,21]
        mask[..., 0] = np.isin(index_matrix, foreground).astype(np.uint8)

        output_cond[:, :, 1] = (fore_index_matrix+1)*28
        output_cond = np.where(mask == 0, 0, output_cond)
        output_cond[:, :, 2] = 127

        return output_cond

    def __getitem__(self, index):
        data_info = self.data_infos[index]
        
        # Read text label
        text = data_info["text"]
        
        # Read images based on mode
        if self.mode == "img2img":
            # Read real_img as target image
            real_img = cv2.imread(data_info["real_img"])
            real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
            real_img = cv2.resize(real_img, (self.resolution, self.resolution))
            real_img = (real_img.astype(np.float32) / 127.5) - 1.0
            
            # Read condition images
            local_conditions = []
            for cond_type in self.local_type_list:
                cond_img = cv2.imread(data_info[cond_type])
                cond_img = cv2.cvtColor(cond_img, cv2.COLOR_BGR2RGB)
                cond_img = cv2.resize(cond_img, (self.resolution, self.resolution))
                cond_img = cond_img.astype(np.float32) / 255.0
                local_conditions.append(cond_img)
            
            # Apply condition dropping
            local_conditions = self.keep_and_drop(local_conditions, self.keep_all_cond_prob, self.drop_all_cond_prob, self.drop_each_cond_prob)
            
            # Concatenate all condition images
            if len(local_conditions) > 0:
                local_conditions = np.concatenate(local_conditions, axis=2)
            else:
                local_conditions = np.zeros((self.resolution, self.resolution, 3), dtype=np.float32)
            
            # Randomly drop text
            if random.random() < self.drop_txt_prob:
                text = 'a driving scenario'
            
            return {
                "jpg": real_img,
                "txt": text,
                "local_conditions": local_conditions,
                "global_conditions": np.zeros((1, 1), dtype=np.float32),  # Not using global conditions
                "mask": np.ones((self.resolution, self.resolution, 1), dtype=np.float32),  # All ones mask
                "name": [f"example_{index}"]
            }
        else:
            # t2i mode, only return text and condition images
            local_conditions = []
            for cond_type in self.local_type_list:
                cond_img = cv2.imread(data_info[cond_type])
                cond_img = cv2.cvtColor(cond_img, cv2.COLOR_BGR2RGB)
                cond_img = cv2.resize(cond_img, (self.resolution, self.resolution))
                cond_img = cond_img.astype(np.float32) / 255.0
                local_conditions.append(cond_img)
                    
            # Concatenate all condition images
            if len(local_conditions) > 0:
                local_conditions = np.concatenate(local_conditions, axis=2)
            else:
                local_conditions = np.zeros((self.resolution, self.resolution, 3), dtype=np.float32)
            
            # Encode conditions
            local_conditions = self.cond_encoder(local_conditions)
            local_conditions = (local_conditions.astype(np.float32) / 127.5) - 1.0
            
            # Randomly drop text
            if random.random() < self.drop_txt_prob:
                text = 'a driving scenario'
            
            return {
                "txt": text,
                "local_conditions": local_conditions,
                "global_conditions": np.zeros((1, 1), dtype=np.float32),  # Not using global conditions
                "mask": np.ones((self.resolution, self.resolution, 1), dtype=np.float32),  # All ones mask
                "name": [f"example_{index}"]
            }

    def __len__(self):
        return len(self.data_infos)

