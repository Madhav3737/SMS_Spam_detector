import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')

def get_pos_tag(pos):
    if pos.startswith('J'):
        return wordnet.ADJ
    elif pos.startswith('V'):
        return wordnet.VERB
    elif pos.startswith('N'):
        return wordnet.NOUN
    elif pos.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN 

def preprocess_msg(msg):
    msg = msg.lower()
    msg = ''.join([c for c in msg if c not in string.punctuation])
    word_list = word_tokenize(msg)
    word_list = [w for w in word_list if w not in stop_words]
    word_pos_tags = pos_tag(word_list)
    word_list = [lemmatizer.lemmatize(w,get_pos_tag(p)) for w,p in word_pos_tags]
    processed_msg = ' '.join(word_list)
    
    return processed_msg