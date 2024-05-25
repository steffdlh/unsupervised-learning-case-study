import pandas as pd
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic




# Load your data
def load_data():
    print(f"Loading data...")
    documents = pd.read_pickle(
        "unsupervised_learning_case_study/data/arxiv-metadata-oai-snapshot-cs.pkl"
    )
    documents = documents[
        pd.to_datetime(documents["update_date"]) > pd.to_datetime("2020-01-01")
    ]
    embeddings = pd.read_pickle("embeddings_SBERT_s.pkl")
    print(f"Len embeddings: {len(embeddings)}")
    print("Finished loading data.")
    
    return documents, embeddings

# Function to extract titles, abstracts, and update dates
def extract_abstracts_dates(documents):
    # pre-filter abstract and update_date null values
    print(f"Pre filter: {documents.shape}")
    documents = documents.dropna(subset=["abstract", "update_date"])
    documents.reset_index(drop=True, inplace=True)
    print(f"Post filter: {documents.shape}")

    # only get first 5000 rows for testing
    abstracts = documents["abstract"].tolist()
    update_dates = pd.to_datetime(documents["update_date"]).tolist()
    print("Finished extracting abstracts and update dates")

    return abstracts, update_dates

# Predict topics for each abstract and return a DataFrame
def predict_topics(embeddings, abstracts, update_dates):
    # Load your BERTopic model
    topic_model: BERTopic = BERTopic.load("topic_model_SBERT.pkl")
    topics, _ = topic_model.transform(abstracts, embeddings)
    df = pd.DataFrame({
        'abstract': abstracts,
        'topic': topics,
        'update_date': update_dates,
        'probabilities': _
    })
    print(df.head(50))
    print("There are {} unique topics".format(len(df['topic'].unique())))
    print(f"Count of -1 topic in df: {len(df[df['topic'] == -1])} which is a percentage of {len(df[df['topic'] == -1])/len(df)*100}%")
    print("Finished predicting topics")
    
    return df

# Main execution function
def main():
    documents, embeddings = load_data()
    abstracts, update_dates = extract_abstracts_dates(documents)
    topics_df = predict_topics(embeddings, abstracts, update_dates)
    
    # Save the dataframe to a CSV file
    topics_df.to_pickle("topics_over_time.pkl")
    print("Dataframe saved to 'topics_over_time.csv'")
    
    return topics_df

# Call the main function
if __name__ == "__main__":
    df = main()
    print(df.head())  # Optionally print the first few rows of the dataframe
