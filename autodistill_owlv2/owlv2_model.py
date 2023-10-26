import os
from dataclasses import dataclass

from PIL import Image
import numpy as np
import torch

import subprocess

import supervision as sv
from autodistill.detection import CaptionOntology, DetectionBaseModel

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class OWLv2(DetectionBaseModel):
    ontology: CaptionOntology
    
    def __init__(self, ontology: CaptionOntology):
        # install transformers from source, since OWLv2 is not yet in a release
        # (as of October 26th, 2023)
        try:
            from transformers import Owlv2Processor, Owlv2ForObjectDetection
        except:
            subprocess.run(["pip3", "install", "git+https://github.com/huggingface/transformers"])
            from transformers import Owlv2Processor, Owlv2ForObjectDetection

        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.ontology = ontology

    def predict(self, input: str, confidence: int = 0.1) -> sv.Detections:
        image = Image.open(input)
        texts = [self.ontology.prompts()]

        inputs = self.processor(text=texts, images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        target_sizes = torch.Tensor([image.size[::-1]])

        results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)

        i = 0
        text = texts[i]

        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

        final_boxes, final_scores, final_labels = [], [], []

        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]

            if score < confidence:
                continue

            final_boxes.append(box)
            final_scores.append(score.item())
            final_labels.append(label.item())
            print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")

        if len(final_boxes) == 0:
            return sv.Detections.empty()
        
        return sv.Detections(
            xyxy=np.array(final_boxes),
            class_id=np.array(final_labels),
            confidence=np.array(final_scores),
        )