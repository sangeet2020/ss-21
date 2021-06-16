from nltk.corpus import reuters, stopwords
stop_words = stopwords.words('english')
# print(reuters.categories())
# print("# of categories: {}".format(len(reuters.categories())))
# reuters.fileids()[:10]

from importlib import reload
import exercise_1
exercise_1 = reload(exercise_1)

corpus = exercise_1.Corpus(
    documents=[exercise_1.Document(fileid, reuters.raw(fileid), reuters.categories(fileid), stop_words=stop_words) for fileid in reuters.fileids()],
    categories=reuters.categories()
)

exercise_1.plot_category_frequencies(corpus.category_frequencies())