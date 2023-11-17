import os
import json
import logging
from typing import List, Dict, Optional, Any

import cv2
from cog import BasePredictor, BaseModel, Input, Path
from autodistill.utils import plot
from autodistill_codet import CoDet
from utils.file_utils import download_model


CHECKPOINT_URLS = [
    ("https://weights.replicate.delivery/default/codet/checkpoints.tar", "/src/CoDet/checkpoints"),
]

class ModelOutput(BaseModel):
    detections: Dict
    result_image: Optional[Path]


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        for (CKPT_URL, target_folder) in CHECKPOINT_URLS:
            if not os.path.exists(target_folder):
                print("Downloading checkpoints and config...")
                download_model(CKPT_URL, target_folder)
        
        self.base_model = CoDet()

    def predict(
        self,
        image: Path = Input(description="Input image"),
        confidence: float = Input(
            description="Confidence threshold to filter detections", 
            ge=0, le=1, default=0.5
        ),
        show_visualisation: bool = Input(description="Plot and show detection results on image", default=True)
    ) -> List[Path]:
        
        """Run a single prediction on the model"""
        predictions, class_names = self.base_model.predict(
            str(image), confidence=confidence
        )

        results = {"objects": []}
        num_dets = predictions.xyxy.shape[0]
        for i in range(num_dets):
            class_id = int(predictions.class_id[i])
            box = [int(i) for i in predictions.xyxy[i].tolist()]
            results["objects"].append({
                "box_xyxy": box,
                "class_id": class_id,
                "class_name": class_names[class_id],
                "confidence": float(predictions.confidence[i])
            })

        file_path = "/tmp/output.json"
        outputs = [Path(file_path)]
        with open(file_path, "w") as outfile: 
            json.dump(results, outfile)

        img_path = "/tmp/output.png"
        result_img = Path(img_path) if show_visualisation else None
        if show_visualisation:
            image = cv2.imread(str(image))
            output = plot(
                image=image, detections=predictions, classes=class_names, raw=True
            )
            cv2.imwrite(img_path, output)
            outputs.append(result_img)

        return outputs