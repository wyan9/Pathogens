import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np

# Load the data
file_path = 'pesticide.xlsx'  # Replace with your file path
data = pd.read_excel(file_path)

# Split the data into features (X) and target (y)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the random forest regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Perform 10-fold cross-validation
cv_scores = cross_val_score(rf, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
cv_rmse = (-cv_scores)**0.5

# Fit the model on the entire training set
rf.fit(X_train, y_train)

# Save the trained model
joblib_file = "Data_forest_model.pkl"
joblib.dump(rf, joblib_file)

# Calculate feature importances
importances = rf.feature_importances_
features = X.columns
feature_importances = pd.DataFrame({'Feature': features, 'Importance': importances})

# Sort feature importances
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 8))
plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()  # Adjust layout
plt.savefig('Data_MF_RF_all_samples.pdf', bbox_inches='tight')  # Ensure saved PDF includes all content
plt.show()

# Display the sorted feature importances
print("Sorted Feature Importances:")
print(feature_importances)

# Predict the target for the test set
y_pred = rf.predict(X_test)

# Calculate the R-squared value
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")

# Initialize SHAP KernelExplainer for smaller subset due to computational intensity
X_sample = X_train.sample(n=100, random_state=42)
explainer = shap.KernelExplainer(rf.predict, X_sample)

# Calculate SHAP values for the test set
shap_values = explainer.shap_values(X_test, nsamples=100)

# Display SHAP summary plot
shap.summary_plot(shap_values, X_test)
