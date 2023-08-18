import pandas as pd
import json
from sklearn.model_selection import train_test_split
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


#jsonline reader function
def jl_reader(input_file):
    with open(input_file) as f:
        lines = f.read().splitlines()
    
    intermediate_df = pd.DataFrame(lines)
    intermediate_df.columns = ['json_element']
    intermediate_df['json_element'].apply(json.loads)
    final_df = pd.json_normalize(intermediate_df['json_element'].apply(json.loads))
    return final_df


# Dataset class to use loader
class GameReviewDataset(Dataset):
    def __init__(self, reviews, numeric_data, labels):
        """
        Args:
        - reviews (List[List[int]]): Tokenized and padded list of reviews. 
                                     Each review is a list of integers (token ids).
        - numeric_data (np.array or torch.Tensor): A 2D array/tensor containing numeric features.
        - labels (List[int]): List of labels.
        """
        self.reviews = reviews
        self.numeric_data = numeric_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.reviews[idx], self.numeric_data[idx], self.labels[idx]

# Define the Neural Network Model
class ReviewClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_numeric_features):
        super(ReviewClassifier, self).__init__()
            
        # Embedding layer for reviews
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            
        # Dense layers
        self.fc1 = nn.Linear(embedding_dim + num_numeric_features, 128)  # embedding_dim for text and num_numeric_features for other numeric data
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Binary classification: Recommended (1) or Not Recommended (0)
            
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_text, x_numeric):
        x_text = self.embedding(x_text)
        x_text = torch.mean(x_text, dim=1)  # Simple Global Average Pooling
            
        # Concatenate text and numeric features
        x = torch.cat((x_text, x_numeric), dim=1)
            
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
            
        return x
    
# Simple tokenizer (split by spaces and create a mapping from word to index)
def build_vocab(reviews):
    vocab = defaultdict(lambda: len(vocab))
    for review in reviews:
        for word in review.split():
            vocab[word]
    return vocab

# Padding to make all reviews same length
def pad_sequences(sequences, maxlen):
    
    padded_seqs = []
    for seq in sequences:
        if len(seq) < maxlen:
            seq = seq + [0] * (maxlen - len(seq))
        else:
            seq = seq[:maxlen]
        padded_seqs.append(seq)
    return padded_seqs


def main():

  # Importing data and preprocessing steps
  # data from all files are concatenated and we remove columns & rows that are of no use.
  #
    files = ['data/Arma_3.jsonlines', 'data/Counter_Strike_Global_Offensive.jsonlines', 'data/Counter_Strike.jsonlines', 
            'data/Dota_2.jsonlines', 'data/Football_Manager_2015.jsonlines', 'data/Garrys_Mod.jsonlines', 
            'data/Grand_Theft_Auto_V.jsonlines', 'data/Sid_Meiers_Civilization_5.jsonlines', 'data/Team_Fortress_2.jsonlines', 
            'data/The_Elder_Scrolls_V.jsonlines', 'data/Warframe.jsonlines']

    # Read data from files and concatenate into a single dataframe
    all_df = pd.concat([jl_reader(file) for file in files], ignore_index=True)

    # Keep only the columns 'review', 'achievement_progress.num_achievements_percentage', and 'total_game_hours'
    all_df = all_df[['review', 'achievement_progress.num_achievements_percentage', 'total_game_hours', 'rating']]

    # Drop rows with NaN in "achievement_progress.num_achievements_percentage" column
    all_df.dropna(subset=['achievement_progress.num_achievements_percentage'], inplace=True)

    
    vocab = build_vocab(all_df['review'])
    # Convert reviews to token ids
    reviews = [[vocab[word] for word in review.split()] for review in all_df['review']]
    MAXLEN = 100
    reviews = pad_sequences(reviews, maxlen=MAXLEN)
    max_token_id = max([max(review) for review in reviews])
   
    
    # Convert 'Recommended' to 1 and 'Not Recommended' to 0
    y = all_df['rating'].replace({'Recommended': 1, 'Not Recommended': 0}).reset_index(drop=True)

    # Split data into training and validation sets
    X_reviews_train, X_reviews_val, X_numeric_train, X_numeric_val, y_train, y_val = train_test_split(
        reviews, all_df[['total_game_hours', 'achievement_progress.num_achievements_percentage']].values, y, test_size=0.2)

    # Convert to PyTorch tensors
    X_reviews_train = torch.tensor(X_reviews_train, dtype=torch.long)
    X_reviews_val = torch.tensor(X_reviews_val, dtype=torch.long)
    X_numeric_train = torch.tensor(X_numeric_train, dtype=torch.float32)
    X_numeric_val = torch.tensor(X_numeric_val, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    y_val = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

    # Create Datasets
    train_dataset = GameReviewDataset(X_reviews_train, X_numeric_train, y_train)
    val_dataset = GameReviewDataset(X_reviews_val, X_numeric_val, y_val)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

     

    # Hyperparameters
    vocab_size = 200000 # Must be greater than highest token_id, which is approx. 150k
    embedding_dim = 128  
    num_numeric_features = 2  # 'total_game_hours' and 'achievement_progress.num_achievements_percentage'
    learning_rate = 0.001
    epochs = 5  

    # Create the model
    model = ReviewClassifier(vocab_size, embedding_dim, num_numeric_features)
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop 
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            text_data, numeric_data, labels = batch
            optimizer.zero_grad()
            outputs = model(text_data, numeric_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            for batch in val_loader:
                text_data, numeric_data, labels = batch
                outputs = model(text_data, numeric_data)
                predicted = torch.round(outputs)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Epoch [{epoch + 1}/{epochs}] - Validation Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    main()