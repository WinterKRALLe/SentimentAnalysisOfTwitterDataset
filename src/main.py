import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
from data_processing import load_data
from topic_modeling import topic_modeling
from sentiment_analyzer import candidate_analysis
from visualization import visualize_frequent_terms, visualize_sentiment_distribution
from text_analysis import analyze_frequent_terms

nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt_tab', quiet=True)

def main():
    stop_words = set(stopwords.words('english'))
    sia = SentimentIntensityAnalyzer()

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)

    df = load_data()
    if df.empty:
        return

    print("Analýza témat:")
    topics = topic_modeling(df)
    print(topics)

    print("\nPredikce sentimentu kandidátů:")
    sentiment_prediction = candidate_analysis(df, sia, stop_words)
    print(sentiment_prediction)

    frequent_terms = analyze_frequent_terms(df, stop_words)
    visualize_frequent_terms(frequent_terms)
    visualize_sentiment_distribution(df, sia)


if __name__ == "__main__":
    main()
