from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

class ImageProcessor:
    def __init__(self, model_name="HuggingFaceM4/idefics2-8b"):
        # Load processor and model
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    
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
# Usage example
if __name__ == "__main__":
    processor = ImageProcessor()
    image_path_base = f'data/meme_images/img{{}}.jpeg'
    image_paths = [image_path_base.format(i) for i in range(1, 12)]
    descriptions = processor.describe_images(image_paths)
    print(descriptions)

