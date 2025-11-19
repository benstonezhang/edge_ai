from abc import ABC, abstractmethod

from torch.nn import Module


class MultiModalModel(ABC):
    vision_mean = [127.5, 127.5, 127.5]
    vision_std = [63.75, 63.75, 63.75]
    image_size = None
    image_token = '<start_of_image>'
    tokenizer_config = {"use_fast": True}
    generation_config = {'max_new_tokens': 1024}
    output_prefix_with_prompt = False

    def __init__(self):
        super().__init__()
        self.generation_model = None

    @classmethod
    @abstractmethod
    def from_pretrained(cls, *args, **kwargs):
        pass

    @abstractmethod
    def get_vision(self) -> Module:
        pass

    @abstractmethod
    def get_input_embeddings(self, processor, text_input: str, image_input: str,
                             audio_input=None, audio_input_mask=None):
        pass

    def eval(self):
        self.generation_model.eval()
        return self

    def generate(self, **inputs):
        return self.generation_model.generate(**inputs)

    def load_image(self, image_path: str):
        from PIL import Image

        orig_image = Image.open(image_path)
        img_width, img_height = self.image_size
        scale_height = img_height / orig_image.height
        scale_width = img_width / orig_image.width
        if scale_height == scale_width:
            image = orig_image.resize((img_width, img_height), reducing_gap=4.0)
        else:
            if scale_height < scale_width:
                scale_size = (min(int(round(orig_image.width * scale_height)), img_width), img_height)
            else:
                scale_size = (img_width, min(int(round(orig_image.height * scale_width)), img_height))
            scaled_image = orig_image.resize(scale_size, reducing_gap=4.0)
            image = Image.new(scaled_image.mode, self.image_size, color=(127, 127, 127))
            image.paste(scaled_image)

        if image.mode != "RGB":
            image = image.convert('RGB')

        return image

    def load_image_tensor(self, image_path: str):
        from torchvision.transforms.functional import pil_to_tensor

        image = self.load_image(image_path)
        pixel_values = pil_to_tensor(image).unsqueeze(0)
        return pixel_values
