import optuna
from transformers import Trainer, TrainingArguments, BertForSequenceClassification
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

# Load ADR dataset and tokenized inputs (replace this part with your actual dataset and tokenized data)
adr_data = pd.read_csv('../data/final_rxnorm_meddra_v2.csv')

# Tokenized inputs (replace with actual tokenization code)
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

# Define the objective function for Optuna
def objective(trial):
    # Hyperparameter search space
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-3)  # Log-uniform distribution for learning rate
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])  # Search over batch sizes
    weight_decay = trial.suggest_uniform('weight_decay', 0.0, 0.1)  # Search for weight decay between 0 and 0.1
    
    # Set up training arguments with the hyperparameters from the trial
    training_args = TrainingArguments(
        output_dir='../outputs/results',          # Output directory for model checkpoints
        num_train_epochs=3,                       # Number of training epochs
        per_device_train_batch_size=batch_size,   # Batch size for training
        per_device_eval_batch_size=batch_size,    # Batch size for evaluation
        evaluation_strategy="epoch",              # Evaluate after every epoch
        save_strategy="epoch",                    # Save the model after every epoch
        logging_dir='../outputs/logs',             # Directory for logs (for TensorBoard)
        logging_steps=10,
        learning_rate=learning_rate,              # Learning rate
        weight_decay=weight_decay,                # Weight decay
        load_best_model_at_end=True,              # Load the best model based on validation performance
        metric_for_best_model="accuracy"         # Monitor accuracy for the best model selection
    )
    
    # Load BioBERT for sequence classification
    model = BertForSequenceClassification.from_pretrained("dmis-lab/biobert-v1.1", num_labels=2)  # Binary classification (2 labels)

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    accuracy = eval_results["eval_accuracy"]
    
    # Return the negative accuracy since Optuna minimizes the objective
    return -accuracy  # We return negative to maximize accuracy in Optuna

# Create an Optuna study and optimize the hyperparameters
study = optuna.create_study(direction='minimize')  # Use 'maximize' for accuracy
study.optimize(objective, n_trials=10)  # Number of trials to perform

# Print the best hyperparameters
print(f"Best hyperparameters: {study.best_trial.params}")