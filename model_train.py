import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load dataset
df = pd.read_csv("Telco-Customer-Churn.csv")

features = ["gender", "MonthlyCharges", "tenure"]
X = df[features]
y = df["Churn"].map({"Yes":1, "No":0})

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessor
categorical = X.select_dtypes(include="object").columns
numeric = X.select_dtypes(exclude="object").columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical)
    ]
)

# Logistic Regression
log_reg = Pipeline(steps=[("preprocessor", preprocessor),
                          ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))])
log_reg.fit(X_train, y_train)

# Decision Tree
dtree = Pipeline(steps=[("preprocessor", preprocessor),
                        ("classifier", DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42))])
dtree.fit(X_train, y_train)

# Save models
joblib.dump(log_reg, "models/log_reg.pkl")
joblib.dump(dtree, "models/decision_tree.pkl")
