import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

# --- 1. Load Dataset ---
# Load the classic Iris dataset
iris = load_iris()
# Convert to DataFrame (X) for better handling
X = pd.DataFrame(iris.data, columns=iris.feature_names)
# Target variable (y)
y = iris.target

# --- Add inspection code here ---
print("\n--- Dataset Inspection ---")
print(f"X shape (rows, columns): {X.shape}") # Shows (150, 4)
print(f"y shape (samples): {y.shape}")     # Shows (150,)
print("\nFirst 5 rows of X (Features):")
print(X.head())
print("\nValue counts of y (Target Species 0, 1, 2):")
print(pd.Series(y).value_counts())
print("--------------------------\n")
# ---------------------------------

# --- 2. Data Preparation ---
# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 3. Model Initialization and Training ---
# CatBoostClassifier for classification problems.
# 'iterations' is the number of trees.
# 'verbose=0' suppresses output during training.
# 'loss_function' for multiclass problem.
model = CatBoostClassifier(
    iterations=100, 
    learning_rate=0.1, 
    depth=6, 
    loss_function='MultiClass', 
    verbose=0,  
    random_seed=42
)

print("Iniciando el entrenamiento del modelo CatBoost en la CPU...")
# Train the model using the fit method
model.fit(X_train, y_train) 
print("¡Entrenamiento completado!")

# --- 4. Prediction and Evaluation ---
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy (y_pred is a 2D array, so we flatten it)
accuracy = accuracy_score(y_test, y_pred.flatten())

print(f"\nPrecisión del modelo en el conjunto de prueba: {accuracy:.4f}")