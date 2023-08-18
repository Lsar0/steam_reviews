import pandas as pd
import json
from sklearn.model_selection import train_test_split
import string
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



# Jsonline reader function
def jl_reader(input_file):
    with open(input_file) as f:
        lines = f.read().splitlines()
    
    intermediate_df = pd.DataFrame(lines)
    intermediate_df.columns = ['json_element']
    intermediate_df['json_element'].apply(json.loads)
    final_df = pd.json_normalize(intermediate_df['json_element'].apply(json.loads))
    return final_df


# Function for text preprocessing
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    return text

# Dataset class for dataloader
class ReviewDataset(Dataset):
    def __init__(self, reviews, hours, achievement_percent, labels):
        self.reviews = reviews
        self.hours = hours
        self.achievement_percent = achievement_percent
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'review': self.reviews[idx],
            'hours': self.hours[idx],
            'achievement_percent': self.achievement_percent[idx],
            'label': self.labels[idx]
        }

# Model
class GameReviewModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_numerical_features):
        super(GameReviewModel, self).__init__()
        
        # Text-related layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # Numerical data layers
        self.dense = nn.Linear(num_numerical_features, 32)  
        self.dropout = nn.Dropout(0.5)
        # Final layers
        self.fc = nn.Linear(hidden_dim + 32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, review, hours, achievement_percent):
        # Text processing
        embedded = self.embedding(review)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1, :]  # Get the last time step's output
        
        # Numerical data processing
        combined_num = torch.cat([hours.unsqueeze(1), achievement_percent.unsqueeze(1)], dim=1)
        dense_out = self.dense(combined_num)
        
        # Combining outputs
        combined = torch.cat([lstm_out, dense_out], dim=1)
        combined = self.dropout(combined)
        
        # Final layers
        out = self.fc(combined)
        return self.sigmoid(out).squeeze()
    
# Function for training and validation loop
def train_model(model, train_loader, val_loader, loss_function, optimizer, num_epochs, device):
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct_train = 0.0

        for batch in train_loader:
            reviews = batch['review'].to(device)
            hours = batch['hours'].to(device)
            achievement_percent = batch['achievement_percent'].to(device)
            labels = batch['label'].to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(reviews, hours, achievement_percent)

            # Calculate the loss
            loss = loss_function(outputs, labels)

            # Backward pass
            loss.backward()

            # Optimize
            optimizer.step()

            total_loss += loss.item()

            # Training accuracy
            predictions = (outputs > 0.5).float()
            correct_train += (predictions == batch['label']).sum().item()


        # Validate the model
        model.eval()
        correct_val = 0
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                reviews = batch['review'].to(device)
                hours = batch['hours'].to(device)
                achievement_percent = batch['achievement_percent'].to(device)
                labels = batch['label'].to(device)

                outputs = model(reviews, hours, achievement_percent)
                val_loss += loss_function(outputs, labels).item()
                predictions = (outputs > 0.5).float()
                correct_val += (predictions == labels).sum().item()

        model.train()

        avg_train_accuracy = correct_train / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_accuracy = correct_val / len(val_loader.dataset)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {total_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.4f}")


# Test evaluation
def evaluate_model(model, test_loader, device):
    model.eval()
    correct_test = 0
    with torch.no_grad():
        for batch in test_loader:
            reviews = batch['review'].to(device)
            hours = batch['hours'].to(device)
            achievement_percent = batch['achievement_percent'].to(device)
            labels = batch['label'].to(device)

            outputs = model(reviews, hours, achievement_percent)
            predictions = (outputs > 0.5).float()
            correct_test += (predictions == labels).sum().item()

    test_accuracy = correct_test / len(test_loader.dataset)
    return test_accuracy


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Importing data and preprocessing steps
    # data from all files are concatenated and we remove columns & rows that are of no use.
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


    # Text preprocessing
    all_df['review'] = all_df['review'].apply(preprocess_text)

    # Tokenizing the reviews using torchtext's tokenizer
    tokenizer = get_tokenizer('basic_english')
    tokenized_reviews = all_df['review'].apply(tokenizer)

    # Building vocabulary from tokenized reviews
    vocab = build_vocab_from_iterator(tokenized_reviews, specials=["<pad>", "<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    # Numericalizing tokenized reviews
    numericalized_reviews = []
    for review in tokenized_reviews:
        numericalized_review = [vocab[token] for token in review]
        numericalized_reviews.append(numericalized_review)

    # Padding sequences to have the same length
    max_length = 100  
    padded_sequences = []
    for seq in numericalized_reviews:
        if len(seq) < max_length:
            seq += [vocab["<pad>"]] * (max_length - len(seq))
        else:
            seq = seq[:max_length]
        padded_sequences.append(seq)

    all_df['padded_sequences'] = padded_sequences
   
    
    # Convert 'Recommended' to 1 and 'Not Recommended' to 0
    y = all_df['rating'].replace({'Recommended': 1, 'Not Recommended': 0}).reset_index(drop=True)

    # Splitting data into train, validation, and test sets
    train_df, temp_df = train_test_split(all_df, test_size=0.2, random_state=99)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=99)

    # Train DataLoader
    train_dataset = ReviewDataset(
        reviews=torch.tensor(train_df['padded_sequences'].tolist(), dtype=torch.long),
        hours=torch.tensor(train_df['total_game_hours'].values, dtype=torch.float32),
        achievement_percent=torch.tensor(train_df['achievement_progress.num_achievements_percentage'].values, dtype=torch.float32),
        labels=torch.tensor(train_df['rating'].replace({'Recommended': 1, 'Not Recommended': 0}).values, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Validation DataLoader
    val_dataset = ReviewDataset(
        reviews=torch.tensor(val_df['padded_sequences'].tolist(), dtype=torch.long),
        hours=torch.tensor(val_df['total_game_hours'].values, dtype=torch.float32),
        achievement_percent=torch.tensor(val_df['achievement_progress.num_achievements_percentage'].values, dtype=torch.float32),
        labels=torch.tensor(val_df['rating'].replace({'Recommended': 1, 'Not Recommended': 0}).values, dtype=torch.float32))

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Test DataLoader
    test_dataset = ReviewDataset(
        reviews=torch.tensor(test_df['padded_sequences'].tolist(), dtype=torch.long),
        hours=torch.tensor(test_df['total_game_hours'].values, dtype=torch.float32),
        achievement_percent=torch.tensor(test_df['achievement_progress.num_achievements_percentage'].values, dtype=torch.float32),
        labels=torch.tensor(test_df['rating'].replace({'Recommended': 1, 'Not Recommended': 0}).values, dtype=torch.float32))

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



    # Creating the model
    model = GameReviewModel(
        vocab_size=len(vocab),
        embedding_dim=50,   
        hidden_dim=100,     
        num_numerical_features=2  # for 'hours' and 'achievement_percent'
    ).to(device)

    
    
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 5
    train_model(model, train_loader, val_loader, loss_function, optimizer, num_epochs, device)

    test_accuracy = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy:.4f}")




if __name__ == "__main__":
    main()