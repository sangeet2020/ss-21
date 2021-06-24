from collections import Counter

from typing import Dict, List


class Document:
    def __init__(self, id:str, text: str, categories: str, stop_words: List[str]):
        self.id = id
        # Determines wheter the document belongs to the train and test set
        self.section = id.split("/")[0]
        # assume only 1 category per document for simplicity
        self.category = categories[0]
        self.stop_words = stop_words
        # TODO: tokenize!
        tokens = None
        # TODO: remove stopwords!
        tokens = None
        # TODO: lemmatize
        lemmatized = None
        # count terms
        self._term_counts = Counter(lemmatized)

    def f(self, term: str) -> int:
        """ returns the frequency of a term in the document """
        return self.term_frequencies[term]

    @property
    def term_frequencies(self):
        return self._term_counts

    @property
    def terms(self):
        return set(self._term_counts.keys())


class Corpus:
    def __init__(self, documents: List[Document], categories: List[str]):
        self.documents = documents
        self.categories = sorted(list(set(categories)))

    def __len__(self):
        return len(self.documents)

    def _tf_idfs(self, document:Document, features:List[str], idfs: Dict[str, float]) -> List[float]:
        raise NotImplementedError

    def _idfs(self, features: List[str]) -> Dict[str, float]:
        raise NotImplementedError

    def _category2index(self, category:str) -> int:
        raise NotImplementedError

    def reduce_vocab(self, min_df: int, max_df: float) -> List[str]:
        raise NotImplementedError

    def compile_dataset(self, reduced_vocab: List[str]) -> Dict:
        raise NotImplementedError

    def category_frequencies(self):
        return Counter([document.category for document in self.documents])

    def terms(self):
        terms = set()
        for document in self.documents:
            terms.update(document.terms)
        return sorted(list(terms))
