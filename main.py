
import argparse
import numpy as np
from src.image_processing_pipeline import ImageProcessingPipeline
from src.embedding_indexer import EmbeddingIndexer
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#ignore warnings
import warnings
warnings.filterwarnings("ignore")

def pipeline_image_descriptions():
    '''
    This function processes images and generates descriptions for them.
    '''
    image_processor = ImageProcessingPipeline()
    image_path_base = f'data/meme_images/img{{}}.jpeg'
    image_paths = [image_path_base.format(i) for i in range(1, 12)]
    descriptions = image_processor.process_images(image_paths)
    
    with open("data/meme_descriptions/descriptions.txt", "w") as f:
        for description in descriptions:
            f.write(description + "\n")

def pipeline_index_dataset():
    '''
    This function indexes text embeddings and saves the index to a file.
    '''
    _ = EmbeddingIndexer(index_path="data/index/index.bin")
    print("Indexing complete")

def pipeline_meme_matcher(query: str, k: int):
    indexer = EmbeddingIndexer(index_path="data/index/index.bin")
    query_emb = indexer.embed_sentences([query])
    similar_sentences, indices = indexer.index_handler.find_similar_sentences(np.stack(query_emb), k)
    print(f"Query: {query}")
    print(f"Most similar meme: {similar_sentences[0]}")
    image_path = f"data/meme_images/img{indices[0][0] + 1}.jpeg"
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.axis('off')  # Optional: Hide axes
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run different pipelines based on the flag passed")
    parser.add_argument("-p", "--pipeline", type=int, help="Pipeline number to run (1 for image descriptions, 2 for indexing dataset, 3 for meme matcher)")
    args = parser.parse_args()
    args.pipeline = 3
    if args.pipeline == 1:
        pipeline_image_descriptions()
    elif args.pipeline == 2:
        pipeline_index_dataset()
    elif args.pipeline == 3:
        query = input("Enter a query: ")
        k = int(input("Enter the number of similar memes to retrieve: "))
        pipeline_meme_matcher(query, k)
    else:
        print("Invalid pipeline number. Please choose 1, 2, or 3.")
