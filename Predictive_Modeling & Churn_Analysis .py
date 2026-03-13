# imports libraries and loads data:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse.linalg.isolve.utils import coerce

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

sns.set(style="whitegrid")

df = pd.read_csv("Cleaned_Preprocessed_Dataset_Week1.csv")

# ------------------------------------------------------------------
# Create Churn Label:

completed_labels = [
    "Completed",
    "Selected",
    "Approved"
]

df["Churn"] = ~df["Status Description"].isin(completed_labels)
df["Apply Date"] = pd.to_datetime(df["Apply Date"], errors="coerce")
df["Entry created at"] = pd.to_datetime(df["Entry created at"], errors="coerce")

df["Churn"] = df["Churn"].astype(int)
df["Churn"] = df["Apply Date"].isna().astype(int)

print("\n++++ Status Description distribution ++++")
print(df["Status Description"].value_counts())

# ------------------------------------------------------------------
# Exploratory Data Analysis:
print("++++ Churn Distribution ++++")
print(df["Churn"].value_counts())
sns.countplot(x='Churn', data=df)
plt.title("Churn vs Retained Students")
plt.show()

# ------------------------------------------------------------------
# Age vs Churn:
print("++++ Age vs Churn ++++")
sns.boxplot(x='Churn', y='Age', data=df)
plt.title("Age Distributed by Churn Status")
plt.show()

# ------------------------------------------------------------------
# Days to Apply vs Churn:
print("++++ Days to Apply vs Churn ++++")
sns.boxplot(x='Churn', y='Days_To_Apply', data=df)
plt.title("Days to Apply by Churn Status")
plt.show()

# ------------------------------------------------------------------
# Gender vs Churn:
print("++++ Gender vs Churn ++++")
sns.countplot(x='Gender', hue='Churn', data=df)
plt.title("Churn by Gender")
plt.show()

# ------------------------------------------------------------------
# Country-wise Churn (Top 10):
print("++++ Country-wise Churn (Top 10) ++++")
top_countries = df['Country'].value_counts().head(10).index
country_churn = df[df['Country'].isin(top_countries)]

plt.figure(figsize=(10,12))
sns.countplot(y='Country', hue='Churn', data=country_churn)
plt.title("Churn by Top Countries")
plt.show()

# ------------------------------------------------------------------
# Predictive Modeling:
print("++++ Feature Selection ++++")
features = ["Age", "Days_To_Apply", "Opportunity_Duration_Days"]

X = df[features]
y = df["Churn"]

# Handle missing & infinite values:
X = X.replace([np.inf, -np.inf], np.nan)

imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\ny_train distribution after split")
print(pd.Series(y_train).value_counts())

# ------------------------------------------------------------------
# Logistic Regression:
print("++++ Logistic Regression ++++")
log_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    solver="liblinear"
)
log_model.fit(X_train_scaled, y_train)

y_pred_log = log_model.predict(X_test_scaled)
print(" ---- Logistic Regression Classification Report ---- ")
print(classification_report(y_test, y_pred_log))

# ------------------------------------------------------------------
# Decision Tree:
print("++++ Decision Tree ++++")
dt_model = DecisionTreeClassifier(max_depth=5)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)
print("      ----- Decision Tree Report -----   ")
print(classification_report(y_test, y_pred_dt))

# ------------------------------------------------------------------
# Random Forest:
print("++++ Random Forest ++++")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
print("       ----- Random Forest Report -----   ")
print(classification_report(y_test, y_pred_rf))

# ------------------------------------------------------------------
# Feature Importance (Random Forest)
importances = pd.Series(
    rf_model.feature_importances_,
    index=features
).sort_values()
plt.figure(figsize=(16,8))
importances.plot(kind='barh')
plt.title("Feature Importance for Churn Prediction")
plt.show()


