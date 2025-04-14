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
                img_dict["sn_depth"] = os.path.join(condition_root, 'sn_depth_front', file_id + '.jpg')
                # img_dict["sn_depth"] = os.path.join('/bigdata/Workspace/Uni-Controlnet/sn_softlinks', 'depth_front', file_id + '.jpg')
                # img_dict["sn_depth"] = os.path.join(ssd_root, 'nuScenes', 'frames', 'CAM_FRONT', file_id + '.jpg')
            if "sn_img" in local_type_list:
                img_dict["sn_img"] = os.path.join(condition_root, 'sn_img_front', file_id + '.jpg')
                # img_dict["sn_img"] = os.path.join('/bigdata/Workspace/Uni-Controlnet/sn_softlinks', 'img_front', file_id + '.jpg')
                # img_dict["sn_img"] = os.path.join(ssd_root, 'nuScenes', 'frames', 'CAM_FRONT', file_id + '.jpg')
            if "sn_seg" in local_type_list:
                img_dict["sn_seg"] = os.path.join(condition_root, 'sn_seg_front', file_id + '.jpg')
                # img_dict["sn_seg"] = os.path.join('/bigdata/Workspace/Uni-Controlnet/sn_softlinks', 'seg_front', file_id + '.jpg')
                # img_dict["sn_seg"] = os.path.join(ssd_root, 'nuScenes', 'annos', 'seg', file_id + '.jpg')

            img_root = "/bigdata/datasets/nuscenes/samples/CAM_FRONT"
            # img_root = os.path.join(ssd_root, 'nuScenes', 'frames', 'CAM_FRONT')
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
                if self.is_train:
                    self.data_infos_ns.append({raw_img: img_dict})
                else:
                    # img_dict["text"] = random.choice((self.annos+self.annos_ytb))
                    self.data_infos_ns.append({raw_img: img_dict})

        self.data_infos_ytb = []
        # for i, file_id in enumerate(file_ids_ytb):
        #     img_dict = {}
        #     # raw_img = os.path.join(ytb_image_dir, file_id + '.jpg')
        #     raw_img = os.path.join(ssd_root, 'YouTube', 'frames', file_id + '.jpg')
        #     if "depth" in local_type_list:
        #         # img_dict["depth"] = os.path.join(condition_root, 'ytb_depth', file_id + '.png')
        #         img_dict["depth"] = os.path.join(ssd_root, 'YouTube', 'annos', 'depth', file_id + '.png')
        #     if "seg" in local_type_list:
        #         # img_dict["seg"] = os.path.join(condition_root, 'ytb_seg', file_id.split("/")[0], "RGB_city", file_id.split("/")[1] + '.jpg')
        #         img_dict["seg"] = os.path.join(ssd_root, 'YouTube', 'annos', 'seg', file_id.split("/")[0], "RGB_city", file_id.split("/")[1] + '.jpg')
        #     if "sn_depth" in local_type_list:
        #         img_dict["sn_depth"]= "BLANK"
        #     if "sn_img" in local_type_list:
        #         img_dict["sn_img"] = "BLANK"
        #     if "sn_seg" in local_type_list:
        #         img_dict["sn_seg"] = "BLANK"

        #     # img_root_ytb = "/bigdata/datasets/YouTube/frames"
        #     img_root_ytb = os.path.join(ssd_root, 'YouTube', 'frames')
        #     index = i + 10

        #     # if index is out of bounds, set it to the last frame
        #     if index >= len(file_ids_ytb):
        #         index = len(file_ids_ytb) - 1

        #     # enforce the image being from the same scene
        #     while file_id.split("/")[0] != file_ids_ytb[index].split("/")[0]:
        #         index -= 1
        #         if index < 0:
        #             index = 0
        #             break
        
        #     img_dict["rgb"] = os.path.join(img_root_ytb, file_ids_ytb[index] + '.jpg')
        #     img_dict["text"] = self.annos_ytb[i]
        #     # if i%ytb_sample_rate == 0:
        #     self.data_infos_ytb.append({raw_img: img_dict})



        # we need to merge the ytb and nuscenes datasets
        print("len(self.image_paths)", len(self.data_infos_ns))
        print("len(self.image_paths_ytb)", len(self.data_infos_ytb))


        # merge
        if self.is_train:
            self.data_infos = self.data_infos_ns*ytb_sample_rate*2 + self.data_infos_ytb*2
            # self.data_infos = self.data_infos_ns
            # self.data_infos = self.data_infos_ytb
        else:
            self.data_infos = self.data_infos_ns
        
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

    def replace_patches(self, image_A, image_B, N=6, rate=0.5):
        height, width = image_A.shape[:2]
        patch_width = width // N
        patch_height = height // N

        for i in range(N):
            for j in range(N):
                patch_A = image_A[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width]
                patch_B = image_B[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width]
                if np.random.random() < rate:
                    image_A[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width] = patch_B

        return image_A

    def cond_encoder(self, image):
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
            [32, 11, 119],       # 19: 'bicycle'
            [189, 176, 55],      # 20: 'cross work'
            [255, 255, 255],     # 21: 'lane line'
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
            [189, 176, 55],      # 20: 'cross work'
            [255, 255, 255],     # 21: 'lane line'
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

    def cond_encoder_continue(self, image):
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
            [32, 11, 119],       # 19: 'bicycle'
            [189, 176, 55],      # 20: 'cross work'
            [255, 255, 255],     # 21: 'lane line'
        ])
        output_cond = image.copy()
        # np.zeros((image.shape[0], image.shape[1], 6), dtype=int)
        # output_cond[:, :, :3] = image[:, :, :3]  # depth
        COLOR_DICT_CITY[:,[2,0]] = COLOR_DICT_CITY[:,[0,2]]
        seg_image = image[:, :, 3:]
        
        colormap = COLOR_DICT_CITY
        colormap_reshaped = colormap.reshape((1, -1, 3))
        distances = np.linalg.norm(seg_image[:, :, np.newaxis, :] - colormap_reshaped, axis=-1)
        index_matrix = np.argmin(distances, axis=-1)

        index_matrix = np.where(index_matrix == 20, 1, index_matrix)
        index_matrix = np.where(index_matrix == 21, 1, index_matrix)

        mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
        foreground = [1,5,12,13,14,15,16,17,18,19,20,21]
        mask[..., 0] = np.isin(index_matrix, foreground).astype(np.uint8)

        # output_cond[:, :, 1] = index_matrix*12
        output_cond = np.where(mask == 0, 0, output_cond)
        # output_cond[:, :, 2] = 127

        return output_cond

    def mix_cond(self, local_cond, global_cond):

        local_cond = ((local_cond +1)*127.5).astype(np.uint8)
        global_cond = ((global_cond +1)*127.5).astype(np.uint8)
        local_cond[:,:,1] = local_cond[:,:,1]//13
        global_cond[:,:,1] = global_cond[:,:,1]//13

        H, W, _ = local_cond.shape
        mask = np.zeros((H, W, 1), dtype=np.uint8)

        foreground = [0,1,11,12,13,14,15,16,17,18]

        mask[..., 0] = np.isin(local_cond[..., 1], foreground).astype(np.uint8)
        local_cond = np.where(mask == 0, 0, local_cond)
        global_cond = np.where(mask == 1, 0, global_cond)

        result = np.maximum(local_cond, global_cond)
        result[:,:,1] = result[:,:,1]*13
        result = (result.astype(np.float32) / 127.5) - 1.0

        return result

    def __getitem__(self, index):
        
        if not self.is_train:
            index = random.randint(0, len(self.data_infos)-1)
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
                # condition = condition.astype(np.float32) / 255.0
                local_conditions.append(condition)
        # if self.is_train and self.condition_rate > 0 and local_conditions[2] is not None and local_conditions[3] is not None:
        #     local_conditions[2] = self.replace_patches(local_conditions[0], local_conditions[2], rate=self.condition_rate)
        #     local_conditions[3] = self.replace_patches(local_conditions[1], local_conditions[3], rate=self.condition_rate)
        # if self.is_train and self.condition_rate > 0 and local_conditions[2] is None and local_conditions[3] is None:
        #     local_conditions[2] = local_conditions[0]
        #     local_conditions[3] = local_conditions[1]
        # # if not self.is_train:
        # local_conditions[0], local_conditions[1] = None, None

        if random.random() < self.drop_txt_prob:
            anno = 'a driving scenario'

        if not self.is_train:
            sn_conditions = np.concatenate(local_conditions[2:4], axis=2)
            sn_conditions = self.cond_encoder(sn_conditions)
            sn_conditions = (sn_conditions.astype(np.float32) / 127.5) - 1.0
        else:
            sn_conditions = np.zeros((self.dsize[1], self.dsize[0], 3), dtype=np.float32)
            # sn_conditions = np.zeros((self.dsize[1], self.dsize[0], 6), dtype=np.float32)
        
        if len(local_conditions) != 0:
            local_conditions = np.concatenate(local_conditions[:2], axis=2)

        local_conditions = self.cond_encoder(local_conditions)
        local_conditions = (local_conditions.astype(np.float32) / 127.5) - 1.0
        # image = local_conditions

        if not self.is_train:
            # sn_conditions = self.mix_cond(sn_conditions, local_conditions)
            sn_conditions = sn_conditions
        else:
            sn_conditions = local_conditions

        return dict(jpg=local_conditions, txt=anno, ori_img=image, sn_img=sn_conditions)
        
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
    local_type_list = ["depth", "seg"]

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
        is_train=False,
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