from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming data.csv has already been loaded and split as before
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base learners
base_learners = [
    ('decision_tree', DecisionTreeClassifier(random_state=42)),
    ('xgboost', XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42))
]

# Define the stacking model with Logistic Regression as the meta-model
stacking_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression(),
    cv=5  # cross-validation for the meta-model's training
)

# Train the stacked model
stacking_model.fit(X_train, y_train)

# Predict and evaluate
stacking_predictions = stacking_model.predict(X_test)
stacking_accuracy = accuracy_score(y_test, stacking_predictions)
print(f"Stacked Model Accuracy: {stacking_accuracy:.4f}")

