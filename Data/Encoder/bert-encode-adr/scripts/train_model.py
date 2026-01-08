import torch
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizer
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.cluster import KMeans

# Load ADR dataset
adr_data = pd.read_csv('../data/final_rxnorm_meddra_v2.csv')

# Tokenized inputs (replace this part with actual tokenized inputs)
tokenized_inputs = torch.load('../outputs/tokenized_inputs.pt')

# Extract labels (replace 'label' with your actual label column)
labels = adr_data['label'].tolist()

# Convert the dataset into a TensorDataset
inputs_ids = tokenized_inputs['input_ids']
attention_mask = tokenized_inputs['attention_mask']
labels_tensor = torch.tensor(labels)

# Create datasets
dataset = TensorDataset(inputs_ids, attention_mask, labels_tensor)
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8)

# Load BioBERT for sequence classification
model = BertForSequenceClassification.from_pretrained("dmis-lab/biobert-v1.1", num_labels=2)  # Binary classification (2 labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir='../outputs/results',            # Output directory for model checkpoints
    num_train_epochs=3,                        # Number of training epochs
    per_device_train_batch_size=8,             # Batch size for training
    per_device_eval_batch_size=8,              # Batch size for evaluation
    evaluation_strategy="epoch",               # Evaluate after every epoch
    save_strategy="epoch",                     # Save the model after every epoch
    logging_dir='../outputs/logs',              # Directory for logs (for TensorBoard)
    logging_steps=10,
    load_best_model_at_end=True,               # Load the best model based on validation performance
    metric_for_best_model="accuracy"          # Monitor accuracy for the best model selection
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained('../outputs/fine_tuned_biobert')
tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
tokenizer.save_pretrained('../outputs/fine_tuned_biobert')

print("Model training completed and saved.")

# Get the embeddings (last hidden state) for the selected ADR texts
with torch.no_grad():
    outputs = model(**tokenized_inputs)

# Extract the [CLS] token embeddings (used as a fixed-length representation)
cls_token_embeddings = outputs.last_hidden_state[:, 0, :]

# Print the shape of the embeddings
print(f"Shape of CLS token embeddings: {cls_token_embeddings.shape}")

# Convert the embeddings to NumPy array for clustering
embeddings = cls_token_embeddings.numpy()

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)  # Set the number of clusters to 5
labels = kmeans.fit_predict(embeddings)

# Display clustering results
for i, label in enumerate(labels):
    print(f"ADR description: {adr_data['meddra_name'][i]} \nAssigned to Cluster: {label}\n")
