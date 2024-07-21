import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch

# Load the feedback data
feedback_df = pd.read_csv('feedback.csv', names=['query', 'response', 'feedback'])

# Create training samples from feedback
train_samples = []
for _, row in feedback_df.iterrows():
    # Use query-response pairs and feedback score (e.g., 1 for positive feedback)
    train_samples.append(InputExample(texts=[row['query'], row['response']], label=float(row['feedback'])))

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Create DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)

# Define loss function
train_loss = losses.CosineSimilarityLoss(model)

# Fine-tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)

# Save the fine-tuned model
model.save('fine_tuned_model')
