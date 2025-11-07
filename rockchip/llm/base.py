from abc import ABC, abstractmethod

from torch.nn import Module
from transformers import AutoProcessor


class MultiModalModel(ABC):
    def __init__(self):
        raise OSError("MultiModalModel is designed to be instantiated "
                      "using the `MultiModalModel.from_pretrained(pretrained_model_name_or_path)` method.")

    @classmethod
    @abstractmethod
    def from_pretrained(cls, *args, **kwargs):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def get_vision(self) -> Module:
        pass

    @property
    def vision_mean(self):
        return [127.5, 127.5, 127.5]

    @property
    def vision_std(self):
        return [127.5, 127.5, 127.5]

    @abstractmethod
    def get_input_embeddings(self, processor: AutoProcessor, text_input: str,
                             image_input=None, audio_input=None, audio_input_mask=None):
        pass
