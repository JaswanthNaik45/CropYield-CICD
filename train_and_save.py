import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ✅ New dataset link (working)
dataset_url = 'https://raw.githubusercontent.com/selva86/datasets/master/Crop_recommendation.csv'

print("Loading dataset...")
df = pd.read_csv(dataset_url)
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("\nFirst 5 rows:\n", df.head())

# Target column is "label" (crop type), so we create a fake numeric target
# Just for experiment replication we simulate yield values
df['Crop_Yield'] = np.random.randint(100, 1000, size=len(df))

# Basic cleaning
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
obj_cols = df.select_dtypes(include=['object']).columns.tolist()

for c in num_cols:
    if df[c].isnull().any():
        df[c].fillna(df[c].median(), inplace=True)
for c in obj_cols:
    if df[c].isnull().any():
        df[c].fillna(df[c].mode()[0], inplace=True)

print("\nMissing values after fill:\n", df.isnull().sum())

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include=['object']):
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop('Crop_Yield', axis=1)
y = df['Crop_Yield']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
}

for name, model in models.items():
    print(f"\nTraining {name} ...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{name}  R²: {r2:.4f}  RMSE: {rmse:.4f}")

# Save XGBoost model
joblib.dump(models["XGBoost"], "model.pkl")
print("\nSaved model to model.pkl")
