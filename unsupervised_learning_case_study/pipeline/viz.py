import pickle
import pandas as pd
import matplotlib.pyplot as plt
from bertopic import BERTopic

# load topics, columns: ['abstract', 'topic', 'update_date', 'probabilities']
topics_df = pd.read_pickle('topics_over_time.pkl')

# drop all rows with topic -1
topics_df = topics_df[topics_df['topic'] != -1]

# load the model
topic_model: BERTopic = BERTopic.load("topic_model_SBERT.pkl")

# remove all rows with missing values
topics_df = topics_df.dropna()

# remove all topics with less than 100 abstracts
topics_df = topics_df.groupby('topic').filter(lambda x: len(x) >= 200)

# remove all topics with having data for less than 12 months
topics_df = topics_df.groupby('topic').filter(lambda x: x['update_date'].dt.to_period('M').nunique() >= 12)

# calculate the 6-month average growth rate for each topic in percent of papers per month
topics_df['month_year'] = topics_df['update_date'].dt.to_period('M').astype(str)
topics_df['growth_rate'] = topics_df.groupby('topic')['month_year'].transform(lambda x: x.value_counts().pct_change().mean() * 100)

# get the last 6 months
last_6_months = topics_df['update_date'].max() - pd.DateOffset(months=12)
top_topics = topics_df[topics_df['update_date'] >= last_6_months].groupby('topic')['growth_rate'].mean().sort_values(ascending=False).head(25).index

# filter the top topics
topics_df = topics_df[topics_df['topic'].isin(top_topics)]

# convert 'update_date' column to datetime format
topics_df['update_date'] = pd.to_datetime(topics_df['update_date'])

# make sure max date is 2024-03-31
topics_df = topics_df[topics_df['update_date'] <= '2024-03-31']


# convert 'month_year' column to string format
topics_df['month_year'] = topics_df['update_date'].dt.to_period('M').astype(str)

# Ensure the 'month_year' column is correctly set up as an index for the plot if needed
grouped = topics_df.groupby(['topic', 'month_year']).size().unstack().fillna(0)

# Create the plot
csv_data = []
for topic in grouped.index:
    fig, ax = plt.subplots(figsize=(15, 10))
    description = topic_model.get_topic(topic)
    description = [word for word, _ in description[:5]]
    ax.plot(grouped.columns, grouped.loc[topic], marker='o', label=f'Topic {topic} - {description}')

    ax.set_title('Abstracts per Month-Year per Topic', fontsize=16)
    ax.set_xlabel('Month-Year', fontsize=12)
    ax.set_ylabel('Count of Abstracts', fontsize=12)
    ax.legend(title='Topics', title_fontsize='13', fontsize='11')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'topic_charts/topics_over_time_{topic}.png')
    plt.close()
    csv_data.append({'topic': topic, 'description': description, 'growth_rate': topics_df['growth_rate'].iloc[topic] , 'filename': f'topics_over_time_{topic}.png'})

# save the csv data
pd.DataFrame(csv_data).to_csv('top_topics_over_time.csv', index=False)

fig = topic_model.visualize_topics()
fig.write_html("distance_map.html")


fig = topic_model.visualize_hierarchy()
fig.write_html("hierarchy.html")