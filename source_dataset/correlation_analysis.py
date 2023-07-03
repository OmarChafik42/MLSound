import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset into a pandas DataFrame
df = pd.read_csv('source_dataset/Spotify_Youtube.csv')

# Calculate the correlation matrix
corr_matrix = df.corr()

# Plot the correlation matrix as a heatmap
sns.heatmap(corr_matrix, cmap='coolwarm')
plt.show()

# Identify pairs of highly correlated features
threshold = 0.8
features_to_remove = []
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            print(f"{corr_matrix.columns[i]} and {corr_matrix.columns[j]} are highly correlated")
            features_to_remove.append(corr_matrix.columns[i])

# Print the list of features to remove
print("Features to remove:", features_to_remove)
