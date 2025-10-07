# download_models.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk

models = [
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "nlptown/bert-base-multilingual-uncased-sentiment",
]

for m in models:
    print("Downloading and caching:", m)
    AutoTokenizer.from_pretrained(m, local_files_only=False)
    AutoModelForSequenceClassification.from_pretrained(m, local_files_only=False)

print("Downloading NLTK data (stopwords, opinion_lexicon, vader)...")
nltk.download("stopwords")
nltk.download("opinion_lexicon")
nltk.download("vader_lexicon")

print("All done. Models and NLTK data are cached locally.")
