import os
import random
import cv2
import numpy as np

from torch.utils.data import Dataset

import sys
sys.path.append('.')
import tqdm

from util import *

ANNO_PATH = "/bigdata/datasets/nuscenes/anno"
SCENE_DATA_PATH  = "/bigdata/datasets/scenarionet/offscreen_render/data_nusc_trainval.json"
TRAIN_SPLIT_PATH = "/bigdata/datasets/nuscenes/splits/train.txt"
VAL_SPLIT_PATH = "/bigdata/datasets/nuscenes/splits/val.txt"

class UniDataset(Dataset):
    def __init__(self,
                 anno_path, # path to base annotation json for nscs images/text prompts, default /bigdata/datasets/nuscenes/anno/text/cam_front.txt
                 ytb_anno_path, # path to base annotation json for ytb images/text prompts, default /home/mlsimon/Uni-ControlNet/all.json
                 image_dir, # path to base image directory for nscs images, default /bigdata/datasets/nuscenes/samples/CAM_FRONT
                 ytb_image_dir, # path to base image directory for ytb images, default /bigdata/datasets/YouTube/frames
                 condition_root, # path to root for all conditions, default /bigdata/datasets/nuscenes/UC-data/conditions_root
                 local_type_list, # list of local conditions, default ["ns_depth_front", "ns_seg_front", " sn_depth_front", "sn_img_front",  "sn_seg_front"]
                 global_type_list, # TODO, but this is just future video frame. Not used for now, default ["rgb"]
                 resolution, # unused for now
                 drop_txt_prob,
                 keep_all_cond_prob,
                 drop_all_cond_prob,
                 drop_each_cond_prob,
                 # NEW PARAMS:
                 scene_data_path=SCENE_DATA_PATH,
                 is_train=True,
                 ytb_sample_rate=20,
                 condition_rate=0.5,
                 ):
        

        ssd_root = '/data1/datasets/'
        self.factor = 2
        self.dsize = (1536//self.factor, 896//self.factor)
        # self.dsize = (512, 256)

        ns_file_ids, self.annos = read_anno_json(anno_path)
        self.image_paths = [os.path.join(image_dir, file_id + '.jpg') for file_id in ns_file_ids]
        self.is_train = is_train
        self.condition_rate = condition_rate

        # with open(TRAIN_SPLIT_PATH, 'rt') as f:
        #     train_file_ids = f.read().splitlines()

        # load scene json to check that future frame is from the same scene
        if self.is_train:
            SPLIT_PATH = TRAIN_SPLIT_PATH
        else:
            SPLIT_PATH = VAL_SPLIT_PATH
        with open(SPLIT_PATH, 'rt') as f:
            train_file_ids = f.read().splitlines()

        with open(SCENE_DATA_PATH, 'rt') as f:
            scene_data = json.load(f)

        # scene_data_split = {key: scene_data[key] for key in train_file_ids if key in scene_data}
        # scene_data = scene_data_split
            
        # NOTE: using all scenes from data_nusc_trainval.json intially, 
        # not considering old trainval split for now
        img_to_scene, scene_img_sets = self.get_img_to_scene(scene_data)

        # filter out images from the test set
        # Note: this is not used for now, as we are filtering train/val/test using train_file_ids
        ns_file_ids_trainval = [file_id for file_id in ns_file_ids if file_id in img_to_scene]
        self.annos = [self.annos[i] for i in range(len(self.annos)) if ns_file_ids[i] in img_to_scene]
        ns_file_ids = ns_file_ids_trainval

        ### START SN
        # sn_root = ANNO_PATH + '/sn'

        # sn_file_ids = []
        # for scene_id in scene_img_sets.keys():
        #     # print(key, infos)
        #     # sn = scenarionet
        #     sn_path = os.path.join(sn_root, scene_id, "CAM_FRONT")
        #     sn_rgb_path = []
        #     for root, ds, fs in os.walk(sn_path):
        #         for f in fs:
        #             if 'rgb_' in f and f.endswith('.jpg'):
        #                 rgb_path = os.path.join(root, f)
        #                 sn_rgb_path.append(rgb_path)
        #     sn_rgb_path = sorted(sn_rgb_path, key=self.extract_number)
        #     sn_rgb_path = self.downsample_10_to_2(sn_rgb_path)
        #     sn_file_ids += sn_rgb_path

        # print("len(sn_rgb_path)", len(sn_file_ids), "len(ns_file_ids)", len(ns_file_ids))

        ### END SN

        ### START YTB
        file_ids_ytb, self.annos_ytb = read_anno_json_ytb(ytb_anno_path)
        self.image_paths_ytb = [os.path.join(ytb_image_dir, file_id + '.jpg') for file_id in file_ids_ytb]

        ### END YTB


        self.data_infos_ns = []
        for i, file_id in enumerate(ns_file_ids):
            img_dict = {}
            raw_img = os.path.join(image_dir, file_id + '.jpg')
            # raw_img = os.path.join(ssd_root, 'nuScenes', 'frames', 'CAM_FRONT', file_id + '.jpg')
            if "depth" in local_type_list:
                img_dict["depth"] = os.path.join(condition_root, 'ns_depth_front', file_id + '.png')
                # img_dict["depth"] = os.path.join(ssd_root, 'nuScenes', 'annos', 'depth', file_id + '.png')
                # img_dict["depth"] ="BLANK"
            if "seg" in local_type_list:
                img_dict["seg"] = os.path.join(condition_root, 'ns_seg_front', file_id + '.jpg')
                # img_dict["seg"] = os.path.join(ssd_root, 'nuScenes', 'annos', 'seg', file_id + '.jpg')
                # img_dict["seg"] = "BLANK"
            if "sn_depth" in local_type_list:
                # img_dict["sn_depth"] = os.path.join(condition_root, 'sn_depth_front', file_id + '.jpg')
                img_dict["sn_depth"] = os.path.join('/bigdata/Workspace/Uni-Controlnet/sn_softlinks', 'depth_front', file_id + '.jpg')
                # img_dict["sn_depth"] = os.path.realpath(img_dict["sn_depth"]).replace('images_updated_box','images_updated_halfbox')
                img_dict["sn_depth"] = os.path.realpath(img_dict["sn_depth"]).replace('images_updated_box','images_updated_model')
                # img_dict["sn_depth"] = os.path.join(ssd_root, 'nuScenes', 'frames', 'CAM_FRONT', file_id + '.jpg')
                # img_dict["sn_depth"] = "BLANK"
            if "sn_img" in local_type_list:
                # img_dict["sn_img"] = os.path.join(condition_root, 'sn_img_front', file_id + '.jpg')
                img_dict["sn_img"] = os.path.join('/bigdata/Workspace/Uni-Controlnet/sn_softlinks', 'img_front', file_id + '.jpg')
                # img_dict["sn_img"] = os.path.realpath(img_dict["sn_img"]).replace('images_updated_box','images_updated_halfbox')
                img_dict["sn_img"] = os.path.realpath(img_dict["sn_img"]).replace('images_updated_box','images_updated_model')
                # img_dict["sn_img"] = os.path.join(ssd_root, 'nuScenes', 'frames', 'CAM_FRONT', file_id + '.jpg')
                # img_dict["sn_img"] = "BLANK"
            if "sn_seg" in local_type_list:
                # img_dict["sn_seg"] = os.path.join(condition_root, 'sn_seg_front', file_id + '.jpg')
                img_dict["sn_seg"] = os.path.join('/bigdata/Workspace/Uni-Controlnet/sn_softlinks', 'seg_front', file_id + '.jpg')
                # img_dict["sn_seg"] = os.path.realpath(img_dict["sn_seg"]).replace('images_updated_box','images_updated_halfbox')
                img_dict["sn_seg"] = os.path.realpath(img_dict["sn_seg"]).replace('images_updated_box','images_updated_model')
                # img_dict["sn_seg"] = os.path.join(ssd_root, 'nuScenes', 'annos', 'seg', file_id + '.jpg')
                # img_dict["sn_seg"] = "BLANK"
            if "sn_ins" in local_type_list:
                img_dict["sn_ins"] = img_dict["sn_seg"].replace('Seg','Instance')

            # img_root = "/bigdata/datasets/nuscenes/samples/CAM_FRONT"
            img_root = os.path.join(ssd_root, 'nuScenes', 'frames', 'CAM_FRONT')
            current_scene = img_to_scene[file_id]
            index = i + 10
            # if index is out of bounds, set it to the last frame
            if index >= len(ns_file_ids):
                index = len(ns_file_ids) - 1
            # enforce the image being from the same scene
            try:
                while img_to_scene[ns_file_ids[index]] != current_scene:
                    # print("scene mismatch, decrementing index")
                    index -= 1
                    if index < 0:
                        index = 0
                        break
            except KeyError:
                print("KeyError:", ns_file_ids[index])
                index = i
            img_dict["rgb"] = os.path.join(img_root, ns_file_ids[index] + '.jpg')
            img_dict["text"] = self.annos[i]
            if current_scene in train_file_ids:
                # if self.is_train:
                self.data_infos_ns.append({raw_img: img_dict})
                # else:
                #     img_dict["text"] = random.choice((self.annos+self.annos_ytb))
                #     self.data_infos_ns.append({raw_img: img_dict})

        self.data_infos_ytb = []
        for i, file_id in enumerate(file_ids_ytb[::400]):
            img_dict = {}
            # raw_img = os.path.join(ytb_image_dir, file_id + '.jpg')
            raw_img = os.path.join(ssd_root, 'YouTube', 'frames', file_id + '.jpg')
            if "depth" in local_type_list:
                # img_dict["depth"] = os.path.join(condition_root, 'ytb_depth', file_id + '.png')
                img_dict["depth"] = os.path.join(ssd_root, 'YouTube', 'annos', 'depth', file_id + '.png')
            if "seg" in local_type_list:
                # img_dict["seg"] = os.path.join(condition_root, 'ytb_seg', file_id.split("/")[0], "RGB_city", file_id.split("/")[1] + '.jpg')
                img_dict["seg"] = os.path.join(ssd_root, 'YouTube', 'annos', 'seg', file_id.split("/")[0], "RGB_city", file_id.split("/")[1] + '.jpg')
            if "sn_depth" in local_type_list:
                img_dict["sn_depth"]= "BLANK"
            if "sn_img" in local_type_list:
                img_dict["sn_img"] = "BLANK"
            if "sn_seg" in local_type_list:
                img_dict["sn_seg"] = "BLANK"
            if "sn_ins" in local_type_list:
                img_dict["sn_ins"] = "BLANK"

            # img_root_ytb = "/bigdata/datasets/YouTube/frames"
            img_root_ytb = os.path.join(ssd_root, 'YouTube', 'frames')
            index = i + 10

            # if index is out of bounds, set it to the last frame
            if index >= len(file_ids_ytb):
                index = len(file_ids_ytb) - 1

            # enforce the image being from the same scene
            while file_id.split("/")[0] != file_ids_ytb[index].split("/")[0]:
                index -= 1
                if index < 0:
                    index = 0
                    break
        
            img_dict["rgb"] = os.path.join(img_root_ytb, file_ids_ytb[index] + '.jpg')
            img_dict["text"] = self.annos_ytb[i]
            # if i%ytb_sample_rate == 0:
            self.data_infos_ytb.append({raw_img: img_dict})

        self.data_infos_ns = self.data_infos_ns[22900:]

        # we need to merge the ytb and nuscenes datasets
        print("len(self.image_paths)", len(self.data_infos_ns))
        print("len(self.image_paths_ytb)", len(self.data_infos_ytb))


        # merge
        # if self.is_train:
        #     self.data_infos = self.data_infos_ns*ytb_sample_rate + self.data_infos_ytb
        #     # self.data_infos = self.data_infos_ns*2
        #     # self.data_infos = self.data_infos_ytb
        # else:
        #     self.data_infos = self.data_infos_ns

        self.data_infos = self.data_infos_ns

        self.data_infos = self.data_infos
        
        # self.image_paths = self.image_paths + self.image_paths_ytb
        # self.annos = self.annos + self.annos_ytb
        # # self.local_paths["depth"] = self.local_paths["depth"] + self.local_paths_ytb["depth"]
        # self.local_paths["depth"] = []
        # for depth_type in [l for l in self.local_paths.keys() if "depth" in l]:
        #     self.local_paths["depth"].extend(self.local_paths[depth_type])

        # self.local_paths["seg"] = []
        # for seg_type in [l for l in self.local_paths.keys() if "seg" in l]:
        #     self.local_paths["seg"].extend(self.local_paths[seg_type])

        # # self.local_paths["seg"] = self.local_paths["seg"] + self.local_paths_ytb["seg"]

        # self.global_paths["rgb"] = self.global_paths["rgb"] + self.global_paths_ytb["rgb"]
        # # end merge

        # print("len(self.image_paths)", len(self.image_paths))


        self.local_type_list = local_type_list
        self.global_type_list = global_type_list
        self.resolution = resolution
        self.drop_txt_prob = drop_txt_prob
        self.keep_all_cond_prob = keep_all_cond_prob
        self.drop_all_cond_prob = drop_all_cond_prob
        self.drop_each_cond_prob = drop_each_cond_prob
    
    def downsample_10_to_2(self, sn_data):
        """
        Takes in a list of scenarionet data and downsamples from 10Hz to 2Hz
        """
        # downsample to 2hz
        new_data = []
        for i in range(0, len(sn_data), 5):
            new_data.append(sn_data[i])

        return new_data

    def extract_number(self, filename):
        return int(filename.split('_')[-1].split('.')[0])

    def get_img_to_scene(self, scene_data):
        # create sets for imgs from each scene, also dict to lookup scene from img
        scene_img_sets = {}
        img_to_scene = {}
        for scene_id in scene_data:
            scene_img_sets[scene_id] = set(scene_data[scene_id]['img_path'])
            for img in scene_data[scene_id]['img_path']:
                img_parsed = img.split('/')[-1].split('.')[0]
                img_to_scene[img_parsed] = scene_id

        return img_to_scene, scene_img_sets

    # def replace_patches(self, image_A, image_B, N=6, rate=0.5):
    #     height, width = image_A.shape[:2]
    #     patch_width = width // N
    #     patch_height = height // N

    #     for i in range(N):
    #         for j in range(N):
    #             patch_A = image_A[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width]
    #             patch_B = image_B[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width]
    #             if np.random.random() < rate:
    #                 image_A[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width] = patch_B

    #     return image_A

    def replace_patches(self, image_A, image_B, N=6, rate=0.5):
        height, width = image_A.shape[:2]
        patch_width = width // N
        patch_height = height // N

        mask = np.random.random((N, N)) < rate
        mask = np.repeat(np.repeat(mask, patch_height, axis=0), patch_width, axis=1)

        image_A[mask] = image_B[mask]

        return image_A

    def add_noise(self, image, mean=0, std_dev=0.05):

        noise = np.random.normal(mean, std_dev, image.shape)
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0, 1)

        return noisy_image
    
    def split_foreground(self, depth_image, seg_image):
        COLOR_DICT_CITY = np.array([
            [128, 64,128],      # 0: 'road'
            [232, 35, 244],     # 1: 'sidewalk'
            [70, 70, 70],       # 2: 'building'
            [156,102,102],      # 3: 'wall'
            [153,153,190],      # 4: 'fence'
            [153,153,153],      # 5: 'pole'
            [30, 170,250],      # 6: 'traffic light'
            [0,220,  220],      # 7: 'traffic sign'
            [35,142, 107],      # 8: 'vegetation'
            [152,251,152],      # 9: 'terrain'
            [180,130,70],       # 10: 'sky'
            [60, 20, 220],      # 11: 'person'
            [0,  0,  255],      # 12: 'rider'
            [142, 0, 0],        # 13: 'car'
            [70,  0, 0],        # 14: 'truck'
            [100, 60,0],        # 15: 'bus'
            [100, 80,0],        # 16: 'train'
            [230, 0, 0],        # 17: 'motorcycle'
            [32, 11, 119],       # 18: 'bicycle'
            [189, 176, 55],      # 19: 'cross work'
            [255, 255, 255]     # 20: 'lane line'
        ])

        if depth_image is None or seg_image is None:
            return [None, None, None, None], None

        fore_depth, back_depth, fore_seg, back_seg = depth_image, depth_image, seg_image, seg_image
        COLOR_DICT_CITY[:,[2,0]] = COLOR_DICT_CITY[:,[0,2]]
        colormap = COLOR_DICT_CITY
        colormap_reshaped = colormap.reshape((1, -1, 3))
        seg_image = (seg_image*255).astype(np.uint8)
        distances = np.linalg.norm(seg_image[:, :, np.newaxis, :] - colormap_reshaped, axis=-1)
        index_matrix = np.argmin(distances, axis=-1)

        mask = np.zeros((seg_image.shape[0], seg_image.shape[1], 1), dtype=np.uint8)
        foreground = [0,4,11,12,13,14,15,16,17,18,19,20]
        mask[..., 0] = np.isin(index_matrix, foreground).astype(np.uint8)

        ori_mask = mask.copy()
        fore_depth = np.where(mask == 0, 0, fore_depth)
        back_depth = np.where(mask == 1, 0, back_depth)
        fore_seg = np.where(mask == 0, 0, fore_seg)
        back_seg = np.where(mask == 1, 0, back_seg)

        # mask[...,0] = np.isin(index_matrix, [19,20]).astype(np.uint8)
        mask = np.logical_or(index_matrix == 19, index_matrix == 20).astype(np.uint8)
        origin_fore_seg = fore_seg.copy()

        fore_seg[mask == 1] = [128/255.0, 64/255.0, 128/255.0]
        # mask = np.repeat(mask, 3, axis=2)
        # mask = mask == 1
        # # Change the color of the corresponding regions in the image matrix
        # fore_seg[mask] = [128/255.0, 64/255.0, 128/255.0]
        # # fore_seg = np.where(mask == 1, 0, fore_seg)

        # # print(index_matrix)
        # print(np.count_nonzero(mask == 1))
        # print(np.count_nonzero(mask == 0))

        # dis = np.sum(np.abs(fore_seg - origin_fore_seg))
        # print(dis)
        # input()

        return [fore_depth, back_depth, fore_seg, back_seg], ori_mask

    def mix_cond(self, back_cond, fore_cond):

        if back_cond is None:
            return fore_cond

        H, W, _ = fore_cond.shape
        mask = np.zeros((H, W, 1), dtype=np.uint8)

        # mask[..., 0] = np.isin(local_cond[..., 1], foreground).astype(np.uint8)
        fore_cond = (fore_cond*255).astype(np.uint8)
        back_cond = (back_cond*255).astype(np.uint8)
        mask[..., 0] = np.isin(fore_cond[..., 0], [0]).astype(np.uint8) & np.isin(fore_cond[..., 1], [0]).astype(np.uint8) & np.isin(fore_cond[..., 2], [0]).astype(np.uint8)
        # mask = np.any(fore_cond != [0, 0, 0], axis=2).astype(np.uint8)[:, :, np.newaxis]

        back_cond = np.where(mask == 0, 0, back_cond)
        return np.maximum(fore_cond,back_cond).astype(np.float32)/255.0
        # return back_cond


    def __getitem__(self, index):
        
        # if not self.is_train:
        #     index = random.randint(0, len(self.data_infos)-1)
        data_info = self.data_infos[index]
        img_path, img_conditions = list(data_info.items())[0]

        # print("img_path", img_path)
        # print("img_conditions", img_conditions["sn_img"])
        # input()

        image = cv2.imread(img_path)
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print("MISSING FILE")
            print(e)
            print(self.image_paths[index])
            exit(1)

        # image = cv2.resize(image, (self.resolution, self.resolution))
        image = cv2.resize(image, dsize=self.dsize, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)

        image = (image.astype(np.float32) / 127.5) - 1.0

        anno = img_conditions["text"]
        local_files = []
        for local_type in self.local_type_list:
            local_files.append(img_conditions[local_type])
        global_files = []
        for global_type in self.global_type_list:
            global_files.append(img_conditions[global_type])

        # flag = "nusc"
        local_conditions = []
        for local_file in local_files:
            if local_file == "BLANK":
                # flag = "ytb"
                # local_conditions.append(np.zeros((self.dsize[1], self.dsize[0], 3), dtype=np.float32))
                local_conditions.append(None)
            else:
                condition = cv2.imread(local_file)
                condition = cv2.cvtColor(condition, cv2.COLOR_BGR2RGB)
                # condition = cv2.resize(condition, (self.resolution, self.resolution))
                condition = cv2.resize(condition, dsize=self.dsize, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
                condition = condition.astype(np.float32) / 255.0
                local_conditions.append(condition)
        # if self.is_train and self.condition_rate > 0 and local_conditions[2] is not None and local_conditions[3] is not None:
        #     local_conditions[2] = self.replace_patches(local_conditions[0], local_conditions[2], rate=self.condition_rate)
        #     local_conditions[3] = self.replace_patches(local_conditions[1], local_conditions[3], rate=self.condition_rate)
        # if self.is_train and self.condition_rate > 0 and local_conditions[2] is None and local_conditions[3] is None:
        #     local_conditions[2] = local_conditions[0]
        #     local_conditions[3] = local_conditions[1]

        # local_conditions: depth, seg, sn_depth, sn_seg, sn_img
        processed_local_conditions = []
        conds, mask_real = self.split_foreground(local_conditions[0], local_conditions[1])
        processed_local_conditions += conds
        conds, mask_sn = self.split_foreground(local_conditions[2], local_conditions[3])
        processed_local_conditions += conds
        mask = mask_sn if mask_sn is not None else mask_real
        # processed_local_conditions += self.split_foreground(local_conditions[0], local_conditions[1])
        # processed_local_conditions += self.split_foreground(local_conditions[2], local_conditions[3])
        for i in range(4, len(local_conditions)):
            processed_local_conditions.append(local_conditions[i])
        assert len(processed_local_conditions) >= 9
        local_conditions = processed_local_conditions # fore_depth, back_depth, fore_seg, back_seg, fore_sn_depth, back_sn_depth, fore_sn_seg, back_sn_seg, sn_img

        del local_conditions[5]
        del local_conditions[6] # remove back_sn_depth and back_sn_seg

        # fore_depth, back_depth, fore_seg, back_seg, fore_sn_depth, fore_sn_seg, sn_img， sn_ins


        # local_conditions[2] = local_conditions[0]
        # local_conditions[3] = local_conditions[1]
        # # if not self.is_train:
        # local_conditions[0], local_conditions[1] = None, None
        
        global_conditions = []
        for global_file in global_files:
            # condition = np.load(global_file)
            condition = np.zeros((self.dsize[1], self.dsize[0], 3), dtype=np.float32)
            # condition = cv2.imread(global_file)
            # condition = cv2.cvtColor(condition, cv2.COLOR_BGR2RGB)
            # condition = cv2.resize(condition, dsize=self.dsize, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
            # condition = condition.astype(np.float32) / 255.0
            global_conditions.append(condition)

        # if random.random() < self.drop_txt_prob:
        #     anno = 'a driving scenario'
        # local_conditions = keep_and_drop(local_conditions, self.keep_all_cond_prob, self.drop_all_cond_prob, self.drop_each_cond_prob)

        # if local_conditions[4] is not None:
        #     if random.random() < 0.35: # 35% chance of not using sn_img
        #         local_conditions[4], local_conditions[5] = None, None
        #     else:
        #         local_conditions[0], local_conditions[2] = None, None
        # if random.random() < 0.5: # 60% chance of not using back_depth and back_seg
        #     local_conditions[1], local_conditions[3] = None, None
        # if random.random() < 0.2 and self.is_train:
        #     local_conditions[6] = None
        # if random.random() < 0.2 and self.is_train:
        #     local_conditions[7] = None

        
        merged_local_conditions = []
        # if local_conditions[4] is not None:
        # if not self.is_train:
        #     merged_local_conditions.append(self.mix_cond(local_conditions[1], local_conditions[4])) # depth
        #     merged_local_conditions.append(self.mix_cond(local_conditions[3], local_conditions[5])) # seg
        # else:
        #     merged_local_conditions.append(self.mix_cond(local_conditions[1], local_conditions[0])) # depth
        #     merged_local_conditions.append(self.mix_cond(local_conditions[3], local_conditions[2])) # seg

        merged_local_conditions.append(self.mix_cond(local_conditions[1], local_conditions[0])) # depth
        merged_local_conditions.append(self.mix_cond(local_conditions[3], local_conditions[2])) # seg

        # merged_local_conditions.append(local_conditions[6]) # sn_img
        # merged_local_conditions.append(local_conditions[7]) # sn_ins
        local_conditions = merged_local_conditions
        

        # final_local_conditions = []
        # if self.is_train:
        #     final_local_conditions = [local_conditions[0], local_conditions[2], local_conditions[1], local_conditions[3], None] # fore_depth, fore_seg, back_depth, back_seg
        # else:
        #     final_local_conditions = [local_conditions[4], local_conditions[5], local_conditions[1], local_conditions[3], None] # fore_sn_depth, fore_sn_seg, back_depth, back_seg
        # local_conditions = final_local_conditions[:2]

        index = random.choice([i for i, x in enumerate(local_conditions) if x is not None])

        mode_rand = random.random()
        model = 'none'

        for i in range(len(local_conditions)):
            if local_conditions[i] is None:
                local_conditions[i] = np.zeros(local_conditions[index].shape, dtype=np.float32)
            else:
                if self.is_train:
                    # if random.random() < 0.4:
                    if model == 'patch' or model == 'both':
                        local_conditions[i] = self.replace_patches(local_conditions[i], np.zeros(local_conditions[index].shape, dtype=np.float32), N=32, rate=0.15)
                    # if random.random() < 0.4:
                    if model == 'noise' or model == 'both':
                        local_conditions[i] = self.add_noise(local_conditions[i], mean=0, std_dev=0.1)
                    # local_conditions[i] = self.add_noise(self.replace_patches(local_conditions[i], np.zeros(local_conditions[index].shape, dtype=np.float32), N=32, rate=0.20))


        global_conditions = keep_and_drop(global_conditions, self.keep_all_cond_prob, self.drop_all_cond_prob, self.drop_each_cond_prob)
        if len(local_conditions) != 0:
            local_conditions = np.concatenate(local_conditions, axis=2)
        if len(global_conditions) != 0:
            global_conditions = np.concatenate(global_conditions)

        return dict(jpg=image, txt=anno, local_conditions=local_conditions, global_conditions=global_conditions, mask=mask, name=img_path)
        
    def __len__(self):
        return len(self.data_infos)
        

if __name__ == '__main__':
    anno_path = "/bigdata/datasets/nuscenes/anno/text/cam_front.txt"
    img_dir = "/bigdata/datasets/nuscenes/samples/CAM_FRONT"
    condition_root = "/bigdata/datasets/nuscenes/UC-data/conditions_root"

    ytb_anno_path = "/bigdata/Workspace/Uni-Controlnet/data/all.json"
    ytb_image_dir = "/bigdata/datasets/YouTube/frames"

    # local_type_list = ["ns_depth_front", "ns_seg_front", "sn_depth_front", "sn_img_front", "sn_seg_front", "ytb_depth", "ytb_seg"]
    local_type_list = ["ns_depth_front", "ns_seg_front", "ytb_depth", "ytb_seg"]
    local_type_list = ["depth", "seg", "sn_depth", "sn_img", "sn_seg"]
    local_type_list = ["depth", "seg", "sn_depth", "sn_img", "sn_seg", "sn_ins"]

    dataset = UniDataset(
        anno_path=anno_path,
        ytb_anno_path=ytb_anno_path,
        image_dir=img_dir,
        ytb_image_dir=ytb_image_dir,
        condition_root=condition_root,
        local_type_list=local_type_list,
        global_type_list=["rgb"],
        resolution=0, # not important
        drop_txt_prob=0.0,
        keep_all_cond_prob=1.0,
        drop_all_cond_prob=0.0,
        drop_each_cond_prob=[0.0 for _ in local_type_list],
        is_train=True,
    )
    
    print(len(dataset))

    test = dataset.__getitem__(1)
    print("image shape:", test["jpg"].shape, "text:", test["txt"], "local conditions [0]shape:", test["local_conditions"].shape, "global conditions, [0]shape:", test["global_conditions"].shape)
    print("\n\n\n")

    for data in dataset:
        # print(data["jpg"].shape, data["txt"], len(data["local_conditions"]), len(data["global_conditions"]))
        print("image shape:", data["jpg"].shape, "text:", data["txt"], "local conditions [0]shape:", data["local_conditions"][0].shape, "global conditions, [0]shape:", data["global_conditions"][0].shape)


    ytb_annos = "/bigdata/datasets/ytb/anno/text/cam_front.txt"
    dataset_ytb = UniDataset()