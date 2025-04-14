import random

import numpy as np


def read_anno(anno_path):
    fi = open(anno_path)
    lines = fi.readlines()
    fi.close()
    file_ids, annos = [], []
    for line in lines:
        id, txt = line.split('\t')
        file_ids.append(id)
        annos.append(txt)
    return file_ids, annos


"""
{
    "/bigdata/datasets/nuscenes/samples/CAM_FRONT/n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915243012465.jpg": "a white van is parked in a parking lot"
}
"""
import json
def read_anno_json(json_path):
    file_ids, annos = [], []
    with open(json_path, 'rt') as f:
        for line in f:
            json_txt = json.loads(line)
            # print(json_txt)
            # there will only be one key-value pair
            for key, value in json_txt.items():
                key_processed = key.split('/')[-1].split('.')[0]
                # print(key_processed)
                file_ids.append(key_processed)
                annos.append(value)
    
    return file_ids, annos


def read_anno_json_ytb(json_path):
    file_ids, annos = [], []
    with open(json_path, 'rt') as f:
        for line in f:
            json_txt = json.loads(line)
            # print(json_txt)
            for key, value in json_txt.items():
                # extracts video parent dir and image path: AEgrx3Uxfts_Singapore 4K - Skyscraper District - Sunset Drive/frame_73515
                key_processed = key.split('/')[-2] + '/' + key.split('/')[-1].split('.')[0]
                file_ids.append(key_processed)
                annos.append(value)
    
    return file_ids, annos


# read_anno_json('/bigdata/datasets/nuscenes/anno/text/cam_front.txt')

def keep_and_drop(conditions, keep_all_prob, drop_all_prob, drop_each_prob):
    results = []
    seed = random.random()
    index = random.choice([i for i, x in enumerate(conditions) if x is not None])
    if seed < keep_all_prob:
        for condition in conditions:
            if condition is None:
                # results.append(np.zeros(conditions[0].shape, dtype=np.float32))
                results.append(np.zeros(conditions[index].shape, dtype=np.float32))
            else:
                results.append(condition)
    elif seed < keep_all_prob + drop_all_prob:
        for condition in conditions:
            # results.append(np.zeros(conditions[0].shape, dtype=np.float32))
            results.append(np.zeros(conditions[index].shape, dtype=np.float32))
    else:
        for i in range(len(conditions)):
            if random.random() < drop_each_prob[i]:
                results.append(None)
            else:
                results.append(conditions[i])
        if all(x is None for x in results):
            chosen_index = random.choice([i for i, x in enumerate(conditions) if x is not None])
            results[chosen_index] = conditions[chosen_index]
        for i in range(len(results)):
            if results[i] is None:
                # results[i] = np.zeros(conditions[0].shape, dtype=np.float32)
                results[i] = np.zeros(conditions[index].shape, dtype=np.float32)
    return results