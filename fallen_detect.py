## model test

import os
import cv2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo

import glob
from tqdm import tqdm

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def save(img, image_path, output=None, dataset_folder="test_data", save_folder="50_373_fallen_result"):
    if output==None:
        save_path = image_path.replace(dataset_folder, save_folder)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img)

    else:
        v = Visualizer(img, metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
        out = v.draw_instance_predictions(output)
        save_path = image_path.replace(dataset_folder, save_folder)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, out.get_image())


# Get the configuration ready
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
predictor = DefaultPredictor(cfg)

# MetadataCatalog.get("keypoints_coco_2017_train").keypoint_names
# keypoint_names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']


keypoint_names = MetadataCatalog.get("keypoints_coco_2017_train").keypoint_names
# _KEYPOINT_THRESHOLD = 0.05
# _RED = (1.0, 0, 0)
# visible = {}


listdir = glob.glob('E:\\data\\mec_data\\save_photo_demo\\image\\*.png')
for imageName in tqdm(listdir):
    new_outputs = False
    img = cv2.imread(imageName)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    outputs = predictor(img)
    if len(outputs["instances"]) == 0:
        save(img, image_path=imageName, output=None)
        continue

    fallens = []
    idx = 0
    for idx in range(len(outputs["instances"])):
        is_fallen = False

        output = outputs["instances"][idx].to("cpu")
        boxes = output.pred_boxes
        x1, y1, x2, y2 = boxes.tensor.numpy()[0]
        aspect_ratio = (y2 - y1) / (x2 - x1)

        if aspect_ratio < 0.667:   # x/y > 1.5
            is_fallen = True
        elif aspect_ratio < 1.25:
            keypoints = output.pred_keypoints
            probs = keypoints[0][:,2]
            left_upper_idx = [0,1,3] # nose, left_eye, left_ear
            right_upper_idx = [0,2,4] # nose, right_eye, right_ear
            left_lower_idx = [13, 15] # left_knee, left_ankle
            right_lower_idx = [14, 16] # right_knee, right_ankle

            left_upper_argmax = left_upper_idx[np.argmax(probs[left_upper_idx])]
            left_upper = keypoints[0][left_upper_argmax]
            left_upper_x, left_upper_y = left_upper[0], left_upper[1]

            right_upper_argmax = right_upper_idx[np.argmax(probs[right_upper_idx])]
            right_upper = keypoints[0][right_upper_argmax]
            right_upper_x, right_upper_y = right_upper[0], right_upper[1]

            left_lower_argmax = left_lower_idx[np.argmax(probs[left_lower_idx])]
            left_lower = keypoints[0][left_lower_argmax]
            left_lower_x, left_lower_y = left_lower[0], left_lower[1]

            right_lower_argmax = right_lower_idx[np.argmax(probs[right_lower_idx])]
            right_lower = keypoints[0][right_lower_argmax]
            right_lower_x, right_lower_y = right_lower[0], right_lower[1]

            left_m = abs((left_upper_y - left_lower_y) / (left_upper_x - left_lower_x))
            right_m = abs((right_upper_y - right_lower_y) / (right_upper_x - right_lower_x))
            # if (aspect_ratio*0.5 < left_m) & (left_m < aspect_ratio*2):
            #     if (aspect_ratio*0.5 < right_m) & (right_m < aspect_ratio*2):
            #         is_fallen = True
            if left_m < 3.73:   # theta = 75
                if right_m < 3.73:
                    is_fallen = True


            # for kp_idx, keypoint in enumerate(keypoints):
            #     x, y, prob = keypoint
            #     if prob > _KEYPOINT_THRESHOLD:
            #         if keypoint_names:
            #             keypoint_name = keypoint_names[kp_idx]
            #             visible[keypoint_name] = (x, y)

        if is_fallen:
            fallens.append(output)
            is_fallen = False


    if len(fallens)>0:
        new_outputs = outputs["instances"].cat(fallens)  # Using instances method 'cat' (not using outputs["instances"])
        save(img=img, image_path=imageName, output=new_outputs)
    else:
        save(img, image_path=imageName, output=None)
