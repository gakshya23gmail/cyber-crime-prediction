# Cyber Crime Prediction using Machine Learning (GitHub Version)

# Required Libraries:
# pandas, scikit-learn

# Importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("../datasets/cyber_crime_data.csv")  # Make sure the dataset is in 'datasets' folder

# Encoding categorical columns
le = LabelEncoder()
for column in ['Gender', 'Location', 'Device', 'Crime']:
    df[column] = le.fit_transform(df[column])

# Features and Target
X = df[['Age', 'Gender', 'Location', 'Device']]
y = df['Crime']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
