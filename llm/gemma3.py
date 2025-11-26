from typing_extensions import override

from .model import MultiModalModel, VisionModelForOnnx


class Gemma3MultiModalModel(MultiModalModel):
    vision_mean = [123.675, 116.28, 103.53]
    vision_std = [58.395, 58.395, 58.395]
    image_size = 896
    # image_tokens = '<start_of_image><image_soft_token>...<end_of_image>'
    generation_config = {'max_new_tokens': 1024, 'do_sample': False}
    output_need_trim = True

    @override
    @classmethod
    def from_pretrained(cls, model_name: str, load_processor: bool, *args, **kwargs):
        from transformers import Gemma3ForConditionalGeneration

        self = cls.__new__(cls)
        self.generation_model = Gemma3ForConditionalGeneration.from_pretrained(model_name, *args, **kwargs)
        self.model = self.generation_model.model if hasattr(self.generation_model, 'model') else self.generation_model
        if load_processor:
            from transformers import Gemma3Processor

            self.processor = Gemma3Processor.from_pretrained(model_name, **self.processor_config)
        return self

    @override
    def get_vision(self):
        return VisionModelForOnnx(self.model)

    @override
    def get_input_embeddings(self, text_input: str, image_input: str, audio_input=None, audio_input_mask=None):
        from PIL import Image

        image = Image.open(image_input)
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_input}
                ]
            }
        ]
        # Example output without token:
        # <bos><start_of_turn>user
        # You are a helpful assistant.
        # <start_of_image> Describe this image.<end_of_turn>
        # <start_of_turn>model
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                    return_dict=True, return_tensors="pt").to(
                self.model.device, dtype=self.model.dtype)
        input_ids = inputs['input_ids']

        # Replace image id with PAD if the image token is OOV, to avoid index-errors
        if self.processor.image_token_id >= self.model.vocab_size:
            special_image_mask = input_ids == self.processor.image_token_id
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids
        inputs_embeds = self.model.get_input_embeddings()(llm_input_ids)

        # Merge text and images
        pixel_values = inputs['pixel_values']
        image_features = self.model.get_image_features(pixel_values).to(inputs_embeds.device, inputs_embeds.dtype)
        if hasattr(self.model, 'get_placeholder_mask'):
            special_image_mask = self.model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds,
                                                                 image_features=image_features)
        else:
            special_image_mask = (input_ids == self.processor.image_token_id).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        return inputs_embeds

    @override
    def generate(self, **inputs):
        generate_ids = self.generation_model.generate(**inputs)
        input_len = inputs["input_ids"].shape[-1]
        return generate_ids[0][input_len:].unsqueeze(0)
