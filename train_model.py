import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

file_path = r"C:\Users\vladb\OneDrive\Documents\punch_features_3piezo.csv"
model_path = r"C:\Users\vladb\OneDrive\Documents\punch_model.pkl"

df = pd.read_csv(file_path)

df = df[df["label"].notna()]
df = df[df["label"] != "unlabeled"]

X = df.drop(columns=["event_id", "label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=None,
    min_samples_split=2,
    random_state=42
)
model.fit(X_train, y_train)
model.feature_names_in_ = X.columns
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print()
print("Classification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(model, model_path)
print(f"\nModel saved to: {model_path}")