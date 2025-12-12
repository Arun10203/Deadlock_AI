import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# 1. Load the dataset
# Ensure 'deadlock_data.csv' is uploaded in the Colab Files tab!
try:
    df = pd.read_csv('deadlock_data.csv')
    print("Dataset loaded successfully.")
    print(df.head())
except FileNotFoundError:
    print("ERROR: 'deadlock_data.csv' not found. Please upload it to the Files tab.")
    exit()

# 2. Preprocessing
# Convert text like "CPU" and "Memory" into numbers (0, 1) so the AI understands.
le = LabelEncoder()
df['Resource_Name_Encoded'] = le.fit_transform(df['Resource_Name'])

# Define Features (Inputs) and Target (Output)
# Inputs: Resource Name (as number), Available Count, Request Amount
X = df[['Resource_Name_Encoded', 'Available', 'Request_Amount']]
# Output: Is_Safe (1 or 0)
y = df['Is_Safe']

# 3. Split into Training and Testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the AI Model (Random Forest)
print("\nTraining the model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# 6. Save the Model and Encoder
# We save these so the Simulator can use them later.
joblib.dump(model, 'deadlock_model.pkl')
joblib.dump(le, 'resource_encoder.pkl')

print("\nSuccess! Files created:")
print("1. deadlock_model.pkl")
print("2. resource_encoder.pkl")
print("Please download these from the Files tab on the left.")
