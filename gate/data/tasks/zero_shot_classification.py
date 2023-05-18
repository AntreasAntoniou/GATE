from typing import Any, List


class ZeroShotViaLabelDescriptionTask:
    def __init__(self, prompt_templates: List = None, label_map: List = None):
        super().__init__()
        self.prompt_templates = prompt_templates
        self.label_map = label_map

    def __call__(self, inputs) -> Any:
        image = inputs["image"]
        labels = inputs["labels"]
        label_description = self.label_map[labels]

        prompts = [
            template.format(label_description)
            for template in self.prompt_templates
        ]

        return {"image": image, "text": prompts, "labels": labels}
