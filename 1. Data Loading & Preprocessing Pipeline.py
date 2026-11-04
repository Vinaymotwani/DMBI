import pandas as pd
import numpy as np
import io
import zipfile
import urllib.request
from sklearn.model_selection import train_test_split

# 1. DATA LOADING & PREPROCESSING PIPELINE (FIXED ZIP HANDLING)
try:
    # Try loading local file first
    df = pd.read_csv('student-mat.csv', sep=';')
    print("Loaded 'student-mat.csv' from local directory.")
except FileNotFoundError:
    print("Local file not found. Fetching directly from UCI ML Repository...")
    # Official UCI zip archive URL
    uci_url = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
    req = urllib.request.Request(uci_url, headers={'User-Agent': 'Mozilla/5.0'})
    
    with urllib.request.urlopen(req) as response:
        parent_zip_bytes = response.read()
    
    # Open parent zip archive
    with zipfile.ZipFile(io.BytesIO(parent_zip_bytes)) as parent_zip:
        # Extract nested student.zip
        nested_zip_bytes = parent_zip.read('student.zip')
        # Open nested student.zip to read student-mat.csv
        with zipfile.ZipFile(io.BytesIO(nested_zip_bytes)) as nested_zip:
            with nested_zip.open('student-mat.csv') as f:
                df = pd.read_csv(f, sep=';')
    
    # Save locally for future executions
    df.to_csv('student-mat.csv', sep=';', index=False)
    print("Successfully unzipped and loaded 'student-mat.csv'!")

# Target Binarization: G3 >= 10 -> Pass (1), G3 < 10 -> At Risk (0)
df['Target'] = (df['G3'] >= 10).astype(int)

# Feature Selection
selected_features = ['studytime', 'failures', 'absences', 'higher', 'romantic', 'G1', 'G2']
X_df = df[selected_features].copy()

# Binary Encoding / One-Hot Encoding
X = pd.get_dummies(X_df, columns=['higher', 'romantic'], drop_first=True)
y = df['Target']

# Stratified Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("\n--- Data Preprocessing Summary ---")
print(f"Total Dataset Records: {len(df)}")
print(f"Training Set Size: {X_train.shape[0]}")
print(f"Testing Set Size: {X_test.shape[0]}")
print("\nTarget Class Breakdown (0 = At Risk, 1 = Pass):")
print(y.value_counts(normalize=True).round(3))
