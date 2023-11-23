from nltk import word_tokenize
from nltk.corpus import stopwords
import string
stop_words=stopwords.words("turkish")
stop_words.extend(["bir","film","filmi","filme","filmde","filmden","filmin","kadar","bi","ben"])

exclude = set(string.punctuation)
def text_process(df):
    token_list = []
    for i, r in df.iterrows():
        text = ''.join(ch for ch in df["lyrics"][i] if ch not in exclude and ch != "’")
        tokens = word_tokenize(text)
        tokens = [tok.lower() for tok in tokens if tok not in stop_words]
        token_list.extend(tokens)
    return token_list