# MemeMatch 🖼️😂
 Welcome to **MemeMatch**, the ultimate tool for finding memes that match your mood or message! Powered by advanced AI technologies like transformer models via Hugging Face and FAISS (Facebook AI Similarity Search), MemeMatch delivers a fun, engaging way to explore memes through their textual descriptions.

## How It Works 🛠️ 
MemeMatch uses three main pipelines to transform meme images into searchable text descriptions, index these descriptions, and then match new queries to the most similar memes:

### 📦 Pipeline 1: 
Build the Dataset This pipeline extracts text descriptions from memes using the [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base) . It processes images stored in data/meme_images and saves the descriptions in data/meme_descriptions/descriptions.txt.

To use this pipeline, run:
```bash
python main.py -p 1 
```

### 🧠 Pipeline 2: 
Embed and Index Descriptions Sentences from data/meme_descriptions/descriptions.txt are embedded using the [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model. These embeddings are then indexed using [FAISS](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/) for efficient similarity searches.

To use this pipeline, run:
```bash
python main.py -p 2
```

### 🔍 Pipeline 3: 
Find Matching Memes This pipeline searches for memes that match a given description. It loads the FAISS index from data/index and retrieves the k-nearest meme descriptions based on the query provided.

To use this pipeline, run:

```bash
python main.py -p 3
``` 

## Real Examples 🌟 

Here are some real-life examples of MemeMatch in action:

Query: "." 
Meme Found: . 

## Getting Started 🚀 
To get started with MemeMatch:
1. Clone this repository. #
2. Install the required dependencies: 
```bash
pip install -r requirements.txt
```
3. Run the desired pipeline as shown above. 
