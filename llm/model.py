from abc import ABC, abstractmethod

from torch.nn import Module


class VisionModelForOnnx(Module):
    def __init__(self, vlm):
        super().__init__()
        self.vlm = vlm
        self.forward = self.vlm.get_image_features


class MultiModalModel(ABC):
    vision_mean = [127.5, 127.5, 127.5]
    vision_std = [63.75, 63.75, 63.75]
    image_size = None
    processor_config = {"use_fast": True}
    generation_config = {'max_new_tokens': 1024}
    onnx_input_names = ['pixel']
    onnx_output_names = ['features']
    onnx_export_conf = {'input_names': onnx_input_names, 'output_names': onnx_output_names}

    def __init__(self):
        super().__init__()
        self.generation_model = None
        self.processor = None

    @classmethod
    @abstractmethod
    def from_pretrained(cls, *args, **kwargs):
        pass

    @abstractmethod
    def get_vision(self) -> Module:
        pass

    @abstractmethod
    def get_input_embeddings(self, text_input: str, image_input: str, audio_input=None, audio_input_mask=None):
        pass

    def eval(self):
        self.generation_model.eval()
        return self

    def generate(self, **inputs):
        return self.generation_model.generate(**inputs)

    def get_image_height_and_width(self):
        import typing

        if self.image_size is None:
            print('Please specify image size')
            exit(1)

        if isinstance(self.image_size, typing.Iterable):
            return self.image_size

        return self.image_size, self.image_size

    def load_image(self, image_path: str, background_color=(127, 127, 127)):
        from PIL import Image

        img_height, img_width = self.get_image_height_and_width()

        orig_image = Image.open(image_path)
        scale_height = img_height / orig_image.height
        scale_width = img_width / orig_image.width
        if scale_height == scale_width:
            image = orig_image.resize((img_width, img_height), reducing_gap=4.0)
        else:
            if scale_height < scale_width:
                width = min(int(round(orig_image.width * scale_height)), img_width)
                scale_size = (width, img_height)
                left_top = ((img_width - width) // 2, 0)
            else:
                height = min(int(round(orig_image.height * scale_width)), img_height)
                scale_size = (img_width, height)
                left_top = (0, (img_height - height) // 2)
            scaled_image = orig_image.resize(scale_size, reducing_gap=4.0)
            image = Image.new(scaled_image.mode, (img_width, img_height), background_color)
            image.paste(scaled_image, left_top)

        if image.mode != "RGB":
            image = image.convert('RGB')

        return image

    def load_image_tensor(self, image_path: str):
        from torchvision.transforms.functional import pil_to_tensor

        image = self.load_image(image_path)
        pixel_values = pil_to_tensor(image).unsqueeze(0)
        return pixel_values
