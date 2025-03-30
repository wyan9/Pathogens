import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# 1. Read the .txt file
file_path = "Pathogens.txt"  # Replace with your file path
df = pd.read_csv(file_path, sep='\t')  # Assuming tab-separated format

# 2. Preview the data (ensure file format is correct)
print("Data preview:")
print(df.head())

# 3. Data normalization (selecting relevant feature columns, assumed to be 'A', 'B', 'C')
features = ['A', 'B', 'C']
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[features] = scaler.fit_transform(df[features])

# 4. Apply K-means clustering (classifying into 3 categories)
kmeans = KMeans(n_clusters=3, random_state=42)
df_scaled['Risk_Level'] = kmeans.fit_predict(df_scaled[features])

# 5. Map risk levels to labels (High, Medium, Low)
risk_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}  # Adjust based on specific needs
df_scaled['Risk_Level_Label'] = df_scaled['Risk_Level'].map(risk_mapping)

# 6. Save results to a file
output_path = "risk_levels_output.txt"
df_scaled.to_csv(output_path, sep='\t', index=False)

# 7. Print partial results
print("Risk level calculation completed! Results saved to:", output_path)
print(df_scaled[['Pathogen_name', 'Risk', 'Risk_Level_Label']].head())
