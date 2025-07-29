import nltk
nltk.data.path.append("./nltk_data")


def ensure_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt',download_dir="./nltk_data")
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger',download_dir="./nltk_data")
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet',download_dir="./nltk_data")
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab',download_dir="./nltk_data")

# Call this during first-time setup

