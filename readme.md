run from data_explorer->vector_search->text_classification-> sentiment_analysis->Dash_board where we can search book follow your description and emotion


#lib you need to download

pandas
numpy
python-dotenv

# LangChain
langchain
langchain-core
langchain-chroma
langchain-huggingface

# Hugging Face & Embeddings
sentence-transformers
transformers
torch

# Vector Database
chromadb

# Package building
setuptools
wheel
build
#models you need to have :
1. Complete AI model list
Model 1 – Emotion Classification
j-hartmann/emotion-english-distilroberta-base
- Classifies book descriptions into 7 emotions.
- Produces emotion scores for recommendation ranking.


Model 2 – Zero-Shot Classification
facebook/bart-large-mnli
- Classifies books into categories without training a dedicated classifier.
- Helps distinguish Fiction, Nonfiction and other categories.


Model 3 – Text Embedding
sentence-transformers/all-MiniLM-L6-v2
- Converts book descriptions and user queries into vectors.
- 384-dimensional embeddings.
- Used with ChromaDB for semantic similarity search.