import json
import os

SCENE_DATA_PATH  = "/bigdata/datasets/scenarionet/offscreen_render/data_nusc_trainval.json"
SN_PATH = "/bigdata/datasets/nuscenes/anno/sn/"

def extract_id(filename):
    return filename.split("/")[-1].split(".")[0]

def extract_id_with_ext(filename):
    return filename.split("/")[-1]

def extract_sn_id(filename):
    return int(filename.split("_")[1].split(".")[0])

def downsample_10_to_2(sn_data):
    """
    Takes in a list of scenarionet data and downsamples from 10Hz to 2Hz
    """
    # downsample to 2hz
    new_data = []
    for i in range(0, len(sn_data), 5):
        new_data.append(sn_data[i])

    return new_data

def downsampling(infos):
    img_path_list = infos['img_path']
    text_list = infos['text']
    img_sweep_list = infos['img_sweep']
    timestamp = infos['timestamp']
    time_interval = infos['time_interval']
    img_sweep_list = sorted(img_sweep_list, key=extract_id)

    new_img_sweep_list = []
    new_timestamp = []
    new_time_interval = []

    i = 0
    while(i < len(time_interval)):
        tmp_img_sweep_list, tmp_timestamp, tmp_time_interval = [], [], []
        one_index = []
        tmp_i = 0
        while (sum(tmp_time_interval)<10 and i+tmp_i < len(time_interval)):
            tmp_img_sweep_list.append(img_sweep_list[i+tmp_i])
            tmp_timestamp.append(timestamp[i+tmp_i])
            tmp_time_interval.append(time_interval[i+tmp_i])
            # print(time_interval[i+tmp_i])
            if time_interval[i+tmp_i] == 1:
                one_index.append(tmp_i)
            tmp_i += 1
        # assert len(tmp_time_interval) >= 5
        # print(one_index, tmp_time_interval)

        i += len(tmp_time_interval)

        if len(tmp_time_interval) <= 5:
            pass
        elif len(tmp_time_interval) == 6:
            del tmp_img_sweep_list[one_index[-1]]
            del tmp_timestamp[one_index[-1]]
            del tmp_time_interval[one_index[-1]]
        else:
            raise ValueError('Error')

        new_img_sweep_list += tmp_img_sweep_list
        new_timestamp += tmp_timestamp
        new_time_interval += tmp_time_interval

    new_text_list = [x for x in text_list for _ in range(5)]
    new_infos = {
        'img_path': img_path_list,
        'text': new_text_list,
        'img_sweep': new_img_sweep_list,
        'timestamp': new_timestamp,
        'time_interval': new_time_interval
    }
    return new_infos

with open(SCENE_DATA_PATH, 'r') as f:
    data = json.load(f)

# # print(extract_id("/bigdata/datasets/nuscenes/samples/CAM_FRONT/n015-2018-07-18-11-07-57+0800__CAM_FRONT__1531883530412470.jpg"))

NEW_JSON_PATH = "/bigdata/datasets/nuscenes/sn_to_nuscenes_tmp.json"

# START CREATE JSON FILE WITH ASSOCIATED IMAGES

new_data = {}
repeat_data = {}

for scene in data:
    # print(scene)
    # imgs = data[scene]['img_path']
    # imgs = data[scene]['img_sweep']
    infos = downsampling(data[scene])
    imgs = infos['img_sweep']
    # print(imgs)

    # imgs = sorted(imgs, key=lambda x: extract_id(x))
    # imgs = [img for i, img in enumerate(imgs) if i % 6 != 3] # remove every 4th image

    # get sn images, depth, seg

    # get all files in os.path.join(SN_PATH, scene) path
    scene_path = os.path.join(SN_PATH, scene)
    # print("scene_path", scene_path)

    cam_path = os.path.join(scene_path, 'CAM_FRONT')
    # print("cam_path", cam_path)
    # depth_path = os.path.join(scene_path, 'Depth_city')
    # seg_path = os.path.join(scene_path, 'Seg_FRONT')
    
    # print(os.listdir(cam_path))
    # print(os.listdir(depth_path))
    # print("seg",os.listdir(seg_path))

    sn_imgs = [img for img in os.listdir(cam_path) if img.endswith('.jpg')]
    # sn_depths = [img for img in os.listdir(depth_path) if img.endswith('.jpg')]
    # sn_segs = [img for img in os.listdir(seg_path) if img.endswith('.jpg')]

    # sort the images by extracting the id and sorting by that
    sn_imgs = sorted(sn_imgs, key=lambda x: extract_sn_id(x))
    # print(sn_imgs)


    # print("lens", len(sn_imgs), len(sn_depths), len(sn_segs))
    # sn_imgs = downsample_10_to_2(sn_imgs)
    # sn_depths = downsample_10_to_2(sn_depths)
    # sn_segs = downsample_10_to_2(sn_segs)
    # print("lens", len(sn_imgs), len(sn_depths), len(sn_segs))
    # print(sn_imgs)


    # now we pair up the sn images with the nuscenes images
    # print(len(sn_imgs), len(imgs))
    # assert len(sn_imgs) == len(imgs)
    length = min(len(sn_imgs), len(imgs))  

    for i in range(length):
        full_path = os.path.join(cam_path, sn_imgs[i])
        new_data[full_path] = imgs[i]
    for i in range(length, len(sn_imgs)):
        full_path = os.path.join(cam_path, sn_imgs[i])
        new_data[full_path] = imgs[-1]
    for i in range(length, len(imgs)):
        full_path = os.path.join(cam_path, sn_imgs[-1])
        repeat_data[imgs[i]] = full_path


with open(NEW_JSON_PATH, 'w') as f:
    json.dump(new_data, f)

## END CREATE JSON FILE WITH ASSOCIATED IMAGES


### START CREATE SOFTLINKS

NEW_IMG_BASE = "/bigdata/Workspace/Uni-Controlnet/sn_softlinks_tmp/img_front"
NEW_DEPTH_BASE = "/bigdata/Workspace/Uni-Controlnet/sn_softlinks_tmp/depth_front"
NEW_SEG_BASE = "/bigdata/Workspace/Uni-Controlnet/sn_softlinks_tmp/seg_front"


# create softlinks for each file in new_data where the softlink title is the value in new_data

with open(NEW_JSON_PATH, 'r') as f:
    new_data = json.load(f)

# for sn_img in new_data:
#     nuscenes_img = new_data[sn_img]

#     # depth_path = sn_img.replace("CAM_FRONT", "Depth_city")
#     # seg_path = sn_img.replace("CAM_FRONT", "Seg_FRONT")
#     depth_path = sn_img.replace("CAM_FRONT", "Depth")
#     seg_path = sn_img.replace("CAM_FRONT", "Seg")
    

#     # print(sn_img, nuscenes_img)

#     # create softlink for sn_img
#     if not os.path.exists(os.path.join(NEW_IMG_BASE, extract_id_with_ext(nuscenes_img))):
#         os.symlink(sn_img, os.path.join(NEW_IMG_BASE, extract_id_with_ext(nuscenes_img)))
#     if not os.path.exists(os.path.join(NEW_DEPTH_BASE, extract_id_with_ext(nuscenes_img))):
#         os.symlink(depth_path, os.path.join(NEW_DEPTH_BASE, extract_id_with_ext(nuscenes_img)))
#     if not os.path.exists(os.path.join(NEW_SEG_BASE, extract_id_with_ext(nuscenes_img))):
#         os.symlink(seg_path, os.path.join(NEW_SEG_BASE, extract_id_with_ext(nuscenes_img)))

for nuscenes_img in repeat_data:
    sn_img = repeat_data[nuscenes_img]

    depth_path = sn_img.replace("CAM_FRONT", "Depth")
    seg_path = sn_img.replace("CAM_FRONT", "Seg")

    # print(sn_img, nuscenes_img)

    # create softlink for sn_img
    if not os.path.exists(os.path.join(NEW_IMG_BASE, extract_id_with_ext(nuscenes_img))):
        os.symlink(sn_img, os.path.join(NEW_IMG_BASE, extract_id_with_ext(nuscenes_img)))
    if not os.path.exists(os.path.join(NEW_DEPTH_BASE, extract_id_with_ext(nuscenes_img))):
        os.symlink(depth_path, os.path.join(NEW_DEPTH_BASE, extract_id_with_ext(nuscenes_img)))
    if not os.path.exists(os.path.join(NEW_SEG_BASE, extract_id_with_ext(nuscenes_img))):
        os.symlink(seg_path, os.path.join(NEW_SEG_BASE, extract_id_with_ext(nuscenes_img)))

### END CREATE SOFTLINKS
