# pip install kagglehub[pandas-datasets]
import kagglehub
import spacy
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split

# Stop words to keep for sentiment analysis
keep_stop_words = {
    "not",
    "no",
    "never",
    "nor",
    "n't",
    "n‘t",
    "n’t",
    "nothing",
    "none",
    "neither",
    "but",
    "very",
    "too",
    "least",
    "against",
    "without",
    "cannot",
}

# n_process spawns workers that re-import this file, so the work must sit behind this guard
if __name__ == "__main__":
    # Load the latest version
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews",
        "IMDB Dataset.csv",
        # kagglehub saves this file still zipped under the .csv name, so pandas has to unzip it
        pandas_kwargs={"compression": "zip"},
    )

    # Remove html tags
    df.review = df.review.str.replace(r"<[^>]+>", " ", regex=True)

    # Preprocess text via spaCy pipeline
    nlp = spacy.load("en_core_web_lg")

    df["tokens"] = [
        [
            # lemma_ keeps the case of proper nouns and "I"; GloVe 6B vocabulary is all lowercase
            r.lemma_.lower()
            for r in review
            if not r.is_punct
            and not r.is_space
            and (not r.is_stop or r.lower_ in keep_stop_words)
        ]
        for review in nlp.pipe(df.review, disable=["parser", "ner"], n_process=4)
    ]

    # Split reviews into train and test data sets
    train, test = train_test_split(
        df, test_size=0.3, random_state=100, stratify=df["sentiment"]
    )
