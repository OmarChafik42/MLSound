import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset into a pandas DataFrame
data = pd.read_csv('datasets/Spotify_YoutubeClean.csv')

# Define the threshold value for the Valence feature
threshold = 0.5

# Create a new column for the class labels
data['mood'] = (data['Valence'] >= threshold).astype(int)

# One-hot encode the categorical variables
data = pd.get_dummies(data)

# Define the input features and target variable
X = data.drop(['mood', 'Valence'], axis=1)
y = data['mood']

# Define the machine learning models to compare
models = {
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier()
}

# Perform cross-validation to compare the performance of the models
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f'{name}: {scores.mean():.3f} (+/- {scores.std():.3f})')
