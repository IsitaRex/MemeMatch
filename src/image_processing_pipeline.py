
from src.image_processor import ImageProcessor

class ImageProcessingPipeline:
    def __init__(self, model_name="HuggingFaceM4/idefics2-8b"):
        self.image_processor = ImageProcessor(model_name=model_name)
    
    def process_images(self, image_paths: list) -> list:
        return self.image_processor.describe_images(image_paths)