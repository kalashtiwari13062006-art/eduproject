
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Project folder ka path
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset load karna
data_path = BASE_DIR / "dataset" / "student-mat.csv"
df = pd.read_csv(data_path, sep=";")

print("Dataset loaded successfully!")
print("Total students:", len(df))
print(df.head())

# Input features
X = df[["studytime", "failures", "absences", "G1", "G2"]]

# Target: final grade
y = df["G3"]

# Training aur testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model train karna
model = LinearRegression()
model.fit(X_train, y_train)

# Model evaluate karna
predictions = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

# Model save karna
model_path = BASE_DIR / "saved_model.pkl"
joblib.dump(model, model_path)

print("Model trained and saved successfully!")
