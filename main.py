import pandas as pd
from src.data_preprocessing import preprocess_data
from src.model import train_model, evaluate_model

# Load data
df = pd.read_csv('data/auto_mpg.csv')

# Preprocess data
X, y = preprocess_data(df)

# Train model
model = train_model(X, y)

# Evaluate model
evaluate_model(model, X, y)
