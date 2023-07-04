import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset into a pandas DataFrame
df = pd.read_csv('source_dataset/Spotify_Youtube.csv')

# Calculate the correlation matrix
corr_matrix = df.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

# Identify pairs of highly correlated features
threshold = 0.8
features_to_remove = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            print(f"{corr_matrix.columns[i]} and {corr_matrix.columns[j]} are highly correlated")
            features_to_remove.add(corr_matrix.columns[i])
            features_to_remove.add(corr_matrix.columns[j])

# Save the plot to an image file
plt.savefig('correlation_matrix.png')

# Print the list of features to remove
print("Features to remove:", list(features_to_remove))
