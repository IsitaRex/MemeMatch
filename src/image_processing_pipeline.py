
from src.image_processor import ImageProcessor

class ImageProcessingPipeline:
    '''
    This class is used to process images and generate descriptions for them.

    Args:
    model_name (str): The name of the pre-trained image captioning model to use. Default is "HuggingFaceM4/idefics2-8b".

    Attributes:
    image_processor (ImageProcessor): The image processor object used to generate descriptions for images.

    Methods:
    process_images: Processes images and generates descriptions for them.
    '''
    def __init__(self, model_name: str ="HuggingFaceM4/idefics2-8b"):
        self.image_processor = ImageProcessor(model_name=model_name)
    
    def process_images(self, image_paths: list) -> list:
        return self.image_processor.describe_images(image_paths)