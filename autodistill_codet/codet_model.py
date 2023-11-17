import os
import sys
import json
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import supervision as sv
from autodistill.detection import DetectionBaseModel
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.modeling import build_model
from detectron2.data.detection_utils import read_image


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autodistill_dir = os.path.expanduser("")
codet_path = os.path.join(autodistill_dir, "CoDet")
center_net_path = os.path.join(codet_path, "third_party/CenterNet2")

sys.path.insert(0, codet_path)
sys.path.insert(0, center_net_path)
from centernet.config import add_centernet_config

deformation_path = os.path.join(codet_path, "third_party/Deformable-DETR")
sys.path.insert(0, deformation_path)
from codet.config import add_codet_config

from autodistill_codet.predictor import VisualizationDemo


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_codet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if '/auto' in cfg.OUTPUT_DIR:
        file_name = os.path.basename(args.config_file)[:-5]
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace('/auto', '/{}'.format(file_name))

    cfg.MODEL.WEIGHTS = "checkpoints/CoDet_OVLVIS_SwinB_4x_ft4x.pth"
    cfg.MODEL.DEVICE = 'cpu'
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

@dataclass
class CoDet(DetectionBaseModel):
    def __init__(self):
        args = SimpleNamespace(
            config_file=os.path.join(codet_path, "configs/CoDet_OVLVIS_SwinB_4x_ft4x.yaml"), 
            resume=False, num_gpus=1, num_machines=1, machine_rank=0, opts=[], eval_only=True,
        )

        # set config_filename
        args.eval_only = True
        args.config_file = os.path.join(codet_path, "configs/CoDet_OVLVIS_SwinB_4x_ft4x.yaml")
        cfg = setup(args)
        os.chdir(codet_path)
        
        # set to cpu
        # self.model = build_model(cfg)
        self.model = build_model(cfg)
        
        DetectionCheckpointer(self.model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        self.cfg = cfg

    def predict(self, input: str, confidence: int = 0.5) -> sv.Detections:
        demo = VisualizationDemo(self.cfg)
        img = read_image(input, format="BGR")
        predictions, _ = demo.run_on_image(img)

        # open autodistill/CoDet/datasets/lvis/lvis_v1_train_norare_cat_info.json
        # map class names to ontology
        class_names = []

        with open("datasets/lvis/lvis_v1_train_norare_cat_info.json") as f:
            data = json.load(f)
            for item in data:
                class_names.append(item["name"])

        predictions = sv.Detections.from_detectron2(predictions)
        predictions = predictions[predictions.confidence > confidence]

        return predictions, class_names