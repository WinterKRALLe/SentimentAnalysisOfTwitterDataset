import matplotlib.pyplot as plt
import seaborn as sns


def visualize_frequent_terms(frequent_terms, candidate=None):
    terms, frequencies = zip(*frequent_terms.items())
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(frequencies), y=list(terms))
    plt.title(f"Top Terms pro {candidate or 'Všechny kandidáty'}")
    plt.xlabel("Frekvence")
    plt.ylabel("Termíny")
    plt.tight_layout()
    plt.savefig("results/frequent_terms.png")


def visualize_sentiment_distribution(df, sia, candidate=None):
    from text_analysis import filter_candidates
    candidate_tweets = filter_candidates(df, candidate)
    sentiments = candidate_tweets['text'].apply(lambda x: sia.polarity_scores(x)['compound'])
    plt.figure(figsize=(10, 6))
    sns.histplot(sentiments, kde=True, bins=30, color='blue')
    plt.title(f"Distribuce sentimentu pro {candidate or 'Všechny kandidáty'}")
    plt.xlabel("Sentiment skóre")
    plt.ylabel("Hustota")
    plt.tight_layout()
    plt.savefig("results/sentiment_distribution.png")
