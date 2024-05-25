from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
from sentence_transformers import SentenceTransformer
import umap.umap_ as UMAP
from hdbscan import HDBSCAN
# from cuml.manifold import UMAP
# from cuml.cluster import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech


def load_data():
    # load arxiv data and filter to only data from 2020 onwards.
    documents = pd.read_pickle(
        "unsupervised_learning_case_study/data/arxiv-metadata-oai-snapshot-cs.pkl"
    )
    documents = documents[
        pd.to_datetime(documents["update_date"]) > pd.to_datetime("2020-01-01")
    ]
    print(f"Finished loading data")
    return documents


def extract_abstracts(documents):
    # Extract titles and abstracts
    abstracts = documents["abstract"].tolist()
    # sentences = [sent_tokenize(abstract) for abstract in abstracts]
    # sentences = [sentence for doc in sentences for sentence in doc]

    # return sentences
    print(f"Finished extracting abstracts")
    return abstracts

def pre_calc_embeddings(abstracts):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(abstracts, show_progress_bar=True)
    print(f"Finished pre-calculating embeddings")
    return embeddings

def run_training(embeddings, abstracts):
    umap_model = UMAP.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=50, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))
    keybert_model = KeyBERTInspired()
    mmr_model = MaximalMarginalRelevance(diversity=0.3)
    pos_model = PartOfSpeech("en_core_web_sm")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    representation_model = {
        "KeyBERT": keybert_model,
        # "OpenAI": openai_model,  # Uncomment if you will use OpenAI
        "MMR": mmr_model,
        "POS": pos_model
    }
    print(f"Starting training")

    topic_model = BERTopic(

        # Pipeline models
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,

        # Hyperparameters
        top_n_words=10,
        verbose=True
    )
    topics, probs = topic_model.fit_transform(abstracts, embeddings)
    # store locally
    pd.to_pickle(topics, 'topics_SBERT.pkl')
    pd.to_pickle(probs, 'probs_SBERT.pkl')
    topic_model.save("topic_model_SBERT.pkl")
    
    print(f"Finished training")
    return topics, probs, topic_model


if __name__ == "__main__":
    documents = load_data()
    sentences = extract_abstracts(documents)
    embeddings = pre_calc_embeddings(sentences)
    pd.to_pickle(embeddings, 'embeddings_SBERT_s.pkl')