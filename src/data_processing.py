import pandas as pd
import re

def load_data():
    df = pd.read_csv("data/Sentiment.csv", encoding='utf-8')
    df.dropna(subset=['text', 'candidate'], inplace=True)
    df['text'] = df['text'].astype(str)
    return df

def advanced_text_cleaning(text):
    text = text.lower()
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()
