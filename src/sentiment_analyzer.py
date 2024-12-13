import pandas as pd
from text_analysis import analyze_frequent_terms, extract_collocations, extract_concordance


def sentiment_analysis(df, sia, candidate=None):
    from text_analysis import filter_candidates
    candidate_tweets = filter_candidates(df, candidate)
    sentiments = candidate_tweets['text'].apply(lambda x: sia.polarity_scores(x)['compound'])
    return {
        'mean_sentiment': float(sentiments.mean()),
        'median_sentiment': float(sentiments.median()),
        'positive_percentage': float((sentiments > 0.05).mean() * 100),
        'neutral_percentage': float(((sentiments >= -0.05) & (sentiments <= 0.05)).mean() * 100),
        'negative_percentage': float((sentiments < -0.05).mean() * 100),
        'total_tweets': int(len(candidate_tweets))
    }


def candidate_analysis(df, sia, stop_words):
    candidates = [c for c in df['candidate'].unique() if c != 'No candidate mentioned'] + ['No candidate mentioned']
    results = []

    keyword = "win"

    for candidate in candidates:

        sentiment_data = sentiment_analysis(df, sia, candidate if candidate != 'No candidate mentioned' else None)
        frequent_terms = analyze_frequent_terms(df, stop_words, candidate if candidate != 'No candidate mentioned' else None)
        collocations = extract_collocations(df, stop_words, candidate if candidate != 'No candidate mentioned' else None)

        # Extrakce konkordancí pro specifického kandidáta
        concordance = extract_concordance(df, keyword, candidate if candidate != 'No candidate mentioned' else None)

        # Formátování konkordancí
        formatted_concordance = [
            f"Context: {''.join(result['left_context'])} **{result['keyword']}** {''.join(result['right_context'])}"
            for result in concordance[:10]
        ]

        results.append({
            'candidate': candidate,
            **sentiment_data,
            'top_terms': list(frequent_terms.keys()),
            'collocations': collocations,
            'concordance': formatted_concordance
        })

    return pd.DataFrame(results)
