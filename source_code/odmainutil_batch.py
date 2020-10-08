import argparse
import glob
import multiprocessing as mp
import os
import cv2
import time
from tqdm import tqdm
import json 
import ffmpeg
from functools import reduce
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import re
import os
from os import listdir
from os.path import isfile, join
from ast import literal_eval
import sys

import base64
import numpy as np
import io
from PIL import Image
# from flask import request
# from flask import jsonify
# from flask import Flask

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"
# basepath = f'/data1/code_base/mnt_data/ODbatch'
# vid_folder = f'{basepath}/vids'
# json_folder = f'{basepath}/JSON'

def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)

    cfg.MODEL.WEIGHTS = '/app/model_final_cafdb1.pkl'
    # cfg.MODEL.WEIGHTS = '/data1/code_base/mnt_data/ODbatch/model_final_cafdb1.pkl'

    # Set score_threshold for builtin models
    #confidence_threshold
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5 
    # confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 

    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5

    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        # default = '/app/docker_files/detectron2/configs/Misc/panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml',
        # default = '/data1/code_base/mnt_data/stream/d2/all_code/docker_files/detectron2/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml',
        default = '/app/docker_files/detectron2/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml',

        # default = f'{basepath}/panoptic_fpn_R_101_3x.yaml',

        metavar="FILE",

        help="path to config file",
    )

    return parser

def load_model():
    args, unknown = get_parser().parse_known_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    model = VisualizationDemo(cfg)

    return model


# Main util for inference

def object_d2(video_id=None, model=None):
    # mp.set_start_method("spawn", force=True)
    
    # for video_id in tqdm(files):
    try:

        #  Load video with CV2
        # video = cv2.VideoCapture(f'{vid_folder}/{video_id}.mp4')
        video = cv2.VideoCapture(f'./{video_id}.mp4')

        print(f'Video name {"<"*10} {video_id}.mp4 >{">"*10} Loaded')

        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        img_pixels = height*width
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))


        print(f'Image height, width, frames_per_second, num_frames and img_pixels is {">"*10}{(height, width, num_frames,frames_per_second, img_pixels)}')

        if frames_per_second ==0:
            pass
        else:
            print(f'video.get(cv2.CAP_PROP_FRAME_COUNT) >> {video.get(cv2.CAP_PROP_FRAME_COUNT)}')

            duration = num_frames/frames_per_second
            
            print('Total frames are ',num_frames)

            frames=[]
            # list of predictions for each frame and object

            all_preds = list(model.run_on_video(video))

            # while num_frames!=0:
                # semantic_predictions = next(all_preds)
                # semantic_predictions = item

            for num_frame, semantic_predictions in enumerate(all_preds):
                objs = []
                for s in semantic_predictions:
                    obj = {}
                    obj["label"] = s["text"]
                    obj['area_percentage'] = float("{0:.2f}".format(s['area']/img_pixels*100))
                    obj["score"] = float("{0:.2f}".format(s["score"] if "score" in s else 1))
                    objs.append(obj)

                obj_set = {}
                for s in semantic_predictions:
                    k = s["text"]
                    score = s["score"] if "score" in s else 1
                    if not k in obj_set:
                        obj_set[k] = {
                            "scores": [score],
                            "areas":  [s["area"]],
                            "label": k
                        }
                    else:
                        obj_set[k]["scores"].append(score)
                        obj_set[k]["areas"].append(s["area"])

                u_objs = []
                for k in obj_set:
                    u = obj_set[k]
                    n = len(u["scores"])
                    score_ave = reduce((lambda x, y: x + y), u["scores"])/n
                    area_sum = reduce((lambda x, y: x + y), u["areas"])

                    obj = {}
                    obj["label"] = u["label"]
                    obj['area_percentage'] = float("{0:.2f}".format(area_sum/img_pixels*100))
                    obj["score"] = float("{0:.2f}".format(score_ave))
                    obj["count"] = n
                    u_objs.append(obj)
                frame = {
                    "frame":num_frame+1,
                    "instances": objs,
                    "objects": u_objs,
                }
                frames.append(frame)
            cv2.destroyAllWindows()
            data = {
                "video": {
                    "meta": {},
                    "base_uri": "https://videobank.blob.core.windows.net/videobank",
                    "folder": video_id,
                    "output-frame-path": ""
                },
                "ml-data": {
                    "object-detection": {
                        "meta": {'duration':duration, 'fps':frames_per_second,'len_frames':len(frames)},
                        "video": {},
                        "frames": frames
                    }
                }
            }

            # print(f'writing OD output inside {">"*10} {json_folder}/{video_id}.json')
            # print(data)
            print(f'{"<"*20} inference completed {">"*20}')
            return data
            # with open(f'{json_folder}/{video_id}.json', 'w') as f:
            #     json.dump(data,f)

    except Exception as e:
        print(f'Caught exception during inference, error is {">"*10} {e}')
        with open(f'/app/err_vidsod.txt','a') as f:
            f.write(str(e))
        pass


# files_ffmpeg=[]
# for f in listdir('{basepath}/ffmpegvid/'):
#     if re.search('\d+\_$',f[:-4]) and os.path.getsize(f'./vids/{f}') != 0:
#         files_ffmpeg.append(f.split('.')[0])

# files = [i.split('.')[0] for i in listdir(f'{vid_folder}')]
# print()f'Total fetched files are {">"*10} {len(files)}'
# files = ['112019_4_']

if __name__=='__main__':
    object_d2(files)
