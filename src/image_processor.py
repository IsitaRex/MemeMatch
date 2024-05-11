from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

class ImageProcessor:
    '''
    This class is used to process images and generate descriptions for them.

    Args:
    model_name (str): The name of the pre-trained image captioning model to use. Default is "Salesforce/blip-image-captioning-base".

    Attributes:
    processor (BlipProcessor): The processor object used to process images.
    model (BlipForConditionalGeneration): The model object used to generate descriptions for images.

    Methods:
    describe_images: Generates descriptions for a list of images.
    '''
    def __init__(self, model_name: str ="Salesforce/blip-image-captioning-base"):
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)

    
    def __describe_image(self, image_path: str) -> str:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
        description = self.processor.decode(outputs[0], skip_special_tokens=True)
        return description
    
    def describe_images(self, image_paths: list) -> list:
        descriptions = []
        for image_path in image_paths:
            description = self.__describe_image(image_path)
            descriptions.append(description)
        return descriptions
