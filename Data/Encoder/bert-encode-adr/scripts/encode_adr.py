import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from datasketch import MinHash, MinHashLSH
import torch
from transformers import BertTokenizer

# Load ADR dataset
adr_data = pd.read_csv('../data/final_rxnorm_meddra_v2.csv')

# Extract ADR texts (we'll use 'meddra_name' column for LSH)
adr_texts = adr_data['meddra_name'].tolist()

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = vectorizer.fit_transform(adr_texts)

# Initialize MinHashLSH (Locality-Sensitive Hashing)
lsh = MinHashLSH(threshold=0.8, num_perm=128)  # Set similarity threshold to 80%

# Create MinHash objects and add them to LSH index
minhashes = []
for i, text in enumerate(adr_texts):
    minhash = MinHash()
    for word in text.split():
        minhash.update(word.encode('utf8'))  # Update the MinHash object with the text words
    lsh.insert(f"text_{i}", minhash)  # Insert the MinHash object into the LSH index with a unique key
    minhashes.append(minhash)

print("LSH index created with MinHash signatures.")

# Example: Search for similar ADR descriptions to the first ADR text
query_text = adr_texts[0]
query_minhash = MinHash()

# Create MinHash for the query text
for word in query_text.split():
    query_minhash.update(word.encode('utf8'))

# Perform similarity search to find similar ADR texts
result = lsh.query(query_minhash)

# Print similar ADR texts (top N results)
print(f"Similar ADR descriptions to '{query_text}':")
for idx in result[:5]:  # Display top 5 similar ADRs
    print(f"- {adr_texts[int(idx.split('_')[1])]}")  # Retrieve text by index

# Load BioBERT tokenizer
tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-v1.1")

# Select the top N similar ADRs for encoding (e.g., top 5)
selected_adr_texts = [adr_texts[int(idx.split('_')[1])] for idx in result[:5]]  # Assuming top 5 results

# Tokenize the selected ADR descriptions
inputs = tokenizer(selected_adr_texts, padding=True, truncation=True, return_tensors="pt")

# Print out the tokenized inputs
print(f"Tokenized {len(selected_adr_texts)} ADR descriptions with BioBERT tokenizer.")