import os
import subprocess
from dataclasses import dataclass
from typing import Any

import numpy as np
import supervision as sv
import torch

from autodistill.detection import CaptionOntology, DetectionBaseModel
from autodistill.helpers import load_image

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class OWLv2(DetectionBaseModel):
    ontology: CaptionOntology

    def __init__(self, ontology: CaptionOntology):
        # install transformers from source, since OWLv2 is not yet in a release
        # (as of October 26th, 2023)
        try:
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
        except:
            subprocess.run(
                ["pip3", "install", "git+https://github.com/huggingface/transformers"]
            )
            from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.processor = Owlv2Processor.from_pretrained(
            "google/owlv2-base-patch16-ensemble"
        )
        self.model = Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-base-patch16-ensemble"
        ).to(DEVICE)  # Move the model to the appropriate device
        self.ontology = ontology

    def predict(self, input: Any, confidence: int = 0.1) -> sv.Detections:
        image = load_image(input, return_format="PIL")
        
        # Check if the image is in RGB format
        if image.mode != "RGB":
            print("Error: Only RGB images are supported for the model")
            return sv.Detections.empty()  # Return an empty detection if the image is not RGB

        texts = [self.ontology.prompts()]

        inputs = self.processor(text=texts, images=image, return_tensors="pt").to(DEVICE)
        outputs = self.model(**inputs)

        target_sizes = torch.Tensor([image.size[::-1]])

        results = self.processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=0.1
        )

        i = 0
        text = texts[i]

        boxes, scores, labels = (
            results[i]["boxes"],
            results[i]["scores"],
            results[i]["labels"],
        )

        final_boxes, final_scores, final_labels = [], [], []

        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]

            if score < confidence:
                continue

            final_boxes.append(box)
            final_scores.append(score.item())
            final_labels.append(label.item())

        if len(final_boxes) == 0:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=np.array(final_boxes),
            class_id=np.array(final_labels),
            confidence=np.array(final_scores),
        )
