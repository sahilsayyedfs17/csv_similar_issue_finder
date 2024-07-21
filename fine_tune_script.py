from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd

# Load your domain-specific dataset
df = pd.read_csv('domain_specific_dataset.csv')

# Prepare data for fine-tuning
train_samples = []
for index, row in df.iterrows():
    train_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=float(row['label'])))

# Create a DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the loss function
train_loss = losses.CosineSimilarityLoss(model)

# Fine-tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)

# Save the fine-tuned model
model.save('fine_tuned_model')
