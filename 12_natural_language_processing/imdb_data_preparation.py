import pandas as pd
import numpy as np
import re
import string
import nltk
import nltk.corpus as nltk_corpus


imdb_reviews = pd.read_csv("data/IMDB Dataset.csv")

# N = 1000
N = len(imdb_reviews)
X = imdb_reviews["review"].iloc[:N]
y = imdb_reviews["sentiment"].iloc[:N]

y.replace({"positive": 1, "negative": 0}, inplace=True)
X = np.array(X)

# lemmatizer = nltk.stem.WordNetLemmatizer()
# stemmer = nltk.stem.PorterStemmer()
stemmer = nltk.stem.LancasterStemmer()

stop_words = nltk_corpus.stopwords.words("english")

for x_row in range(len(X)):
    X[x_row] = re.sub("<.*?>", " ", X[x_row])
    X[x_row] = X[x_row].lower()
    X[x_row] = X[x_row].translate(str.maketrans("", "", string.punctuation))
    
    X[x_row] = nltk.word_tokenize(X[x_row])
    
    # X[x_row] = [lemmatizer.lemmatize(word) for word in X[x_row]]
    X[x_row] = [stemmer.stem(word) for word in X[x_row]]
    
    # X[x_row] = [word for word in X[x_row] if word not in stop_words]
    
    X[x_row] = " ".join(X[x_row])

    if x_row % 100 == 0:
        print("{}/{} reviews prepared.".format(x_row, len(X)))
else:
    print("{}/{} reviews prepared.".format(len(X), len(X)))

imdb_reviews["review"] = X
imdb_reviews["sentiment"] = y

imdb_reviews.to_csv("data/imdb_dataset_prepared.csv", index=False)
