from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.text import Text
from data_processing import advanced_text_cleaning

def tokenize(text, stop_words):
    tokens = text.split()
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    return tokens

def extract_ngrams(text, n=2):
    tokens = word_tokenize(text)
    return [' '.join(gram) for gram in ngrams(tokens, n)]

def analyze_frequent_terms(df, stop_words, candidate=None):
    from data_processing import advanced_text_cleaning
    candidate_tweets = filter_candidates(df, candidate)
    all_tokens = []
    for text in candidate_tweets['text']:
        cleaned_text = advanced_text_cleaning(text)
        all_tokens.extend(tokenize(cleaned_text, stop_words))
    return dict(Counter(all_tokens).most_common(10))

def filter_candidates(df, candidate=None):
    if candidate:
        return df[df['candidate'] == candidate]
    return df[df['candidate'] != 'No candidate mentioned']

def extract_collocations(df, stop_words, candidate=None, n=2):
    """Extract collocations for a candidate."""
    candidate_tweets = filter_candidates(df, candidate)
    all_tokens = []
    for text in candidate_tweets['text']:
        cleaned_text = advanced_text_cleaning(text)
        all_tokens.extend(tokenize(cleaned_text, stop_words))
    ngram_counts = Counter(ngrams(all_tokens, n))
    return [f"{' '.join(gram)} ({count})" for gram, count in ngram_counts.most_common(10)]



def extract_concordance(df, keyword, candidate=None, window=5):
    concordance_results = []
    candidate_tweets = filter_candidates(df, candidate)
    
    for text in candidate_tweets['text']:
        # Preprocessing
        cleaned_text = advanced_text_cleaning(text)
        tokens = word_tokenize(cleaned_text.lower())
        
        # Najdi všechny výskyty klíčového slova
        keyword_indices = [i for i, token in enumerate(tokens) if token == keyword.lower()]
        
        for idx in keyword_indices:
            # Extrahuj kontext
            start = max(0, idx - window)
            end = min(len(tokens), idx + window + 1)
            
            context = {
                'full_tweet': text,
                'left_context': ' '.join(tokens[start:idx]),
                'keyword': tokens[idx],
                'right_context': ' '.join(tokens[idx+1:end]),
                'candidate': candidate if candidate else 'Všichni'
            }
            
            concordance_results.append(context)
    
    return concordance_results
