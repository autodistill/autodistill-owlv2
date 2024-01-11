import os
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import supervision as sv
import torch
from autodistill.detection import CaptionOntology, DetectionBaseModel
from autodistill.helpers import load_image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class OWLv2(DetectionBaseModel):
    ontology: CaptionOntology

    def __init__(
        self,
        ontology: CaptionOntology,
        model: Optional[Union[str, os.PathLike]] = "google/owlv2-base-patch16-ensemble",
    ):
        self.ontology = ontology
        self.processor = Owlv2Processor.from_pretrained(model)
        self.model = Owlv2ForObjectDetection.from_pretrained(model).to(DEVICE)

    def predict(self, input: Any, confidence: int = 0.1) -> sv.Detections:
        texts = [self.ontology.prompts()]

        image = load_image(input, return_format="PIL")

        with torch.no_grad():
            inputs = self.processor(text=texts, images=image, return_tensors="pt").to(
                DEVICE
            )
            outputs = self.model(**inputs)

            # Model bb output is on padded square preprocessed image. We need to adjust target_sizes
            # accordingly.
            max_dim = max(image.size)
            target_sizes = torch.Tensor([[max_dim, max_dim]])

            results = self.processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes
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
