
import argparse
from src.image_processing_pipeline import ImageProcessingPipeline

def pipeline1():
    image_processor = ImageProcessingPipeline()
    image_path_base = f'data/meme_images/img{{}}.jpeg'
    image_paths = [image_path_base.format(i) for i in range(1, 12)]
    descriptions = image_processor.process_images(image_paths)
    
    # save the descriptions to a file data/meme_descriptions/descriptions.txt
    with open("data/meme_descriptions/descriptions.txt", "w") as f:
        for description in descriptions:
            f.write(description + "\n")

def pipeline2():
    pass

def pipeline3():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run different pipelines based on the flag passed")
    parser.add_argument("-p", "--pipeline", type=int, help="Pipeline number to run (1, 2, or 3)")
    args = parser.parse_args()

    if args.pipeline == 1:
        pipeline1()
    elif args.pipeline == 2:
        pipeline2()
    elif args.pipeline == 3:
        pipeline3()
    else:
        print("Invalid pipeline number. Please choose 1, 2, or 3.")
