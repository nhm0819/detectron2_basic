# check pytorch installation:
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# Some basic setup:
# Setup detectron2 logger
import detectron2

from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from MyTrainer import MyTrainer

dataset = 'person'

# path
data_path = os.path.join(os.getcwd(), 'datasets')
server_path = '/mnt/data/COCO/coco2017'

mec_train_path = os.path.join(server_path, 'train2017')
mec_val_path = os.path.join(server_path, 'val2017')
mec_train_json = os.path.join(data_path, 'person_keypoints_train2017.json')
mec_val_json = os.path.join(data_path, 'person_keypoints_val2017.json')


# keypoint names, mapping
# keypoint_names = [
#     'nose',
#     'left_eye',
#     'right_eye',
#     'left_ear',
#     'right_ear',
#     'left_shoulder',
#     'right_shoulder',
#     'left_elbow',
#     'right_elbow',
#     'left_wrist',
#     'right_wrist',
#     'left_hip',
#     'right_hip',
#     'left_knee',
#     'right_knee',
#     'left_ankle',
#     'right_ankle'
# ]
keypoint_names = MetadataCatalog.get("keypoints_coco_2017_train").keypoint_names
keypoint_flip_map = {
    'left_eye': 'right_eye',
    'left_ear': 'right_ear',
    'left_shoulder': 'right_shoulder',
    'left_elbow': 'right_elbow',
    'left_wrist': 'right_wrist',
    'left_hip': 'right_hip',
    'left_knee': 'right_knee',
    'left_ankle': 'right_ankle'
}


# Detectron2
register_coco_instances("mec_dataset_train", {}, mec_train_json, mec_train_path)
register_coco_instances("mec_dataset_val", {}, mec_val_json, mec_val_path)

MetadataCatalog.get("mec_dataset_train").keypoint_names = keypoint_names
MetadataCatalog.get("mec_dataset_train").keypoint_flip_map = keypoint_flip_map
MetadataCatalog.get("mec_dataset_val").keypoint_names = keypoint_names
MetadataCatalog.get("mec_dataset_val").keypoint_flip_map = keypoint_flip_map

MetadataCatalog.get("mec_dataset_train").set(thing_classes = ["person"])
MetadataCatalog.get("mec_dataset_val").set(thing_classes = ["person"])

### Hyperparameters
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("mec_dataset_train", )
cfg.DATASETS.TEST = ("mec_dataset_val", )
cfg.TEST.EVAL_PERIOD = 10000
cfg.DATALOADER.NUM_WORKERS = 16
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
cfg.SOLVER.MAX_ITER =  250000 - 1
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # num classes (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)



### Training
cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, '{}_1'.format(dataset))
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()