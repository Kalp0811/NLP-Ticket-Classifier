# src/explore_data.py

import pandas as pd
from collections import Counter
from . import config

def prepare_and_sample_data():
    """
    Loads the full dataset, preprocesses it, and creates a stratified sample.
    """
    print("Loading full dataset...")
    df = pd.read_csv(config.FULL_DATA_PATH)

    # 1. Map Ticket Types to our target labels
    df['label'] = df['Ticket Type'].map(config.TICKET_TYPE_MAP).fillna('Other')

    # 2. Combine Subject and Description for a richer text feature
    df['text'] = df['Ticket Subject'] + ". " + df['Ticket Description']
    
    # 3. Perform Stratified Sampling to maintain label distribution
    print(f"Performing stratified sampling for {config.SAMPLE_SIZE} examples...")
    df_sampled = df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(int(len(x) * config.SAMPLE_SIZE / len(df)))
    )
    
    # Ensure we get exactly SAMPLE_SIZE, adjusting for rounding
    if len(df_sampled) < config.SAMPLE_SIZE:
        # Add a few more random samples to meet the size requirement
        additional_samples = df.drop(df_sampled.index).sample(config.SAMPLE_SIZE - len(df_sampled))
        df_sampled = pd.concat([df_sampled, additional_samples])

    # 4. Save the sampled dataset
    df_sampled[['text', 'label']].to_csv(config.SAMPLED_DATA_PATH, index=False)
    print(f"✅ Sampled data saved to {config.SAMPLED_DATA_PATH}")
    
    return df_sampled

def run_eda(df):
    """
    Performs Exploratory Data Analysis on the provided dataframe.
    """
    print("\n--- Starting Exploratory Data Analysis (EDA) ---")

    # 1. Label Distribution
    print("\n📊 Label Distribution:")
    label_counts = df['label'].value_counts()
    print(label_counts)

    # 2. Text Length Analysis
    print("\n📝 Text Length Analysis (in characters):")
    df['text_length'] = df['text'].str.len()
    print(df['text_length'].describe())

    # 3. N-gram Analysis (Top 3 for each category)
    print("\n🔍 Top N-grams per Category:")
    for label in config.LABELS:
        print(f"\n--- Category: {label} ---")
        text = " ".join(df[df['label'] == label]['text'])
        words = text.lower().split()
        
        # Unigrams
        unigrams = Counter(words).most_common(3)
        print(f"Top Unigrams: {unigrams}")
        
        # Bigrams
        bigrams = Counter(zip(words, words[1:])).most_common(3)
        print(f"Top Bigrams: {bigrams}")

    print("\n✅ EDA Complete.")


if __name__ == "__main__":
    sampled_df = prepare_and_sample_data()
    run_eda(sampled_df)