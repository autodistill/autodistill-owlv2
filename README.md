<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png"
      >
    </a>
  </p>
</div>

# Autodistill OWLv2 Module

This repository contains the code supporting the OWLv2 base model for use with [Autodistill](https://github.com/autodistill/autodistill).

OWLv2 is a zero-shot object detection model that follows from on the OWL-ViT architecture. OWLv2 has an open vocabulary, which means you can provide arbitrary text prompts for the model. You can use OWLv2 with autodistill for object detection.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [OWLv2 Autodistill documentation](https://autodistill.github.io/autodistill/base_models/owlv2/).

## Installation

To use OWLv2 with autodistill, you need to install the following dependency:


```bash
pip3 install autodistill-owlv2
```

## Quickstart

```python
from autodistill_owlv2 import OWLv2

# define an ontology to map class names to our OWLv2 prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = OWLv2(
    ontology=CaptionOntology(
        {
            "person": "person",
            "a forklift": "forklift"
        }
    )
)
base_model.label("./context_images", extension=".jpeg")
```


## License

This model is licensed under an [Apache 2.0](LICENSE) ([see original model implementation license](https://huggingface.co/docs/transformers/main/en/model_doc/owlv2), and the corresponding [HuggingFace Transformers documentation](https://huggingface.co/docs/transformers/main/en/model_doc/owlv2)).

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!