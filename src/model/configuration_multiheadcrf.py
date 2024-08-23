
from transformers import PretrainedConfig, AutoConfig
from typing import List


class MultiHeadCRFConfig(PretrainedConfig):
    model_type = "crf-tagger"

    def __init__(
        self,
        classes = list(),
        number_of_layer_per_head = 1,
        augmentation = "random",
        context_size = 64,
        percentage_tags = 0.2,
        aug_prob = 0.5,
        crf_reduction = "mean",
        freeze = False,
        version="0.1.3",
        **kwargs,
    ):
        self.classes = classes
        self.number_of_layer_per_head=number_of_layer_per_head
        self.version = version
        self.augmentation = augmentation
        self.context_size = context_size
        self.percentage_tags = percentage_tags
        self.aug_prob = aug_prob,
        self.crf_reduction = crf_reduction
        self.freeze=freeze
        self.version = version
        super().__init__(**kwargs)
        

        