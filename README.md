# CatBoost Iris Classifier 🌸

A machine learning project that uses **CatBoost** to classify iris flower species based on their physical characteristics. This implementation demonstrates gradient boosting for multiclass classification on the classic Iris dataset.

## 📋 Description

This project implements a CatBoost classifier to predict iris species (Setosa, Versicolor, and Virginica) based on four features:
- Sepal length
- Sepal width
- Petal length
- Petal width

The model achieves high accuracy using gradient boosting techniques with minimal code and configuration.

## 🚀 Features

- **Dataset Loading**: Uses the built-in Iris dataset from scikit-learn
- **Data Inspection**: Displays dataset shape, features, and class distribution
- **Train/Test Split**: 80/20 split for model validation
- **CatBoost Implementation**: Gradient boosting classifier with multiclass support
- **Performance Evaluation**: Accuracy metrics on test data

## 📦 Requirements

```bash
pandas
scikit-learn
catboost
```

## 🔧 Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/catboost-iris-classifier.git
cd catboost-iris-classifier
```

2. Install dependencies:
```bash
pip install pandas scikit-learn catboost
```

## 💻 Usage

Run the script:
```bash
python iris_classifier.py
```

Expected output:
```
--- Dataset Inspection ---
X shape (rows, columns): (150, 4)
y shape (samples): (150,)

First 5 rows of X (Features):
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                5.1               3.5                1.4               0.2
1                4.9               3.0                1.4               0.2
...

Value counts of y (Target Species 0, 1, 2):
0    50
1    50
2    50
--------------------------

Iniciando el entrenamiento del modelo CatBoost en la CPU...
¡Entrenamiento completado!

Precisión del modelo en el conjunto de prueba: 1.0000
```

## 🧠 Model Architecture

**CatBoost Classifier Parameters:**
- `iterations`: 100 trees
- `learning_rate`: 0.1
- `depth`: 6 (tree depth)
- `loss_function`: MultiClass (for 3 species)
- `random_seed`: 42 (for reproducibility)

## 📊 Dataset Information

**Iris Dataset:**
- **Samples**: 150 (50 per species)
- **Features**: 4 numeric attributes
- **Classes**: 3 (Setosa, Versicolor, Virginica)
- **Split**: 120 training samples, 30 test samples

## 🎯 Performance

The model typically achieves **~97-100% accuracy** on the test set, demonstrating CatBoost's effectiveness for this classification task.

## 🔍 How It Works

1. **Data Loading**: Imports the Iris dataset and converts to pandas DataFrame
2. **Data Inspection**: Prints dataset dimensions and distribution
3. **Data Splitting**: Divides data into training (80%) and testing (20%) sets
4. **Model Training**: Trains CatBoost classifier with 100 iterations
5. **Prediction**: Makes predictions on unseen test data
6. **Evaluation**: Calculates and displays accuracy score

## 🌟 Why CatBoost?

- **High Performance**: Often outperforms other gradient boosting libraries
- **Easy to Use**: Minimal hyperparameter tuning required
- **Built-in Categorical Support**: Handles categorical features automatically
- **Fast Training**: Optimized for CPU training
- **Robust**: Less prone to overfitting

## 📚 Learning Objectives

This project helps you understand:
- Gradient boosting for classification
- CatBoost library implementation
- Multiclass classification problems
- Model evaluation techniques
- Data preprocessing with pandas

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Add new features
- Improve documentation

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

Your Name - [@yourhandle](https://twitter.com/yourhandle)

## 🙏 Acknowledgments

- Built with [CatBoost](https://catboost.ai/)
- Dataset from [scikit-learn](https://scikit-learn.org/)
- Inspired by gradient boosting techniques

## 📖 Further Reading

- [CatBoost Documentation](https://catboost.ai/docs/)
- [Iris Dataset Information](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset)
- [Gradient Boosting Explained](https://en.wikipedia.org/wiki/Gradient_boosting)

---

⭐ If you found this helpful, please give it a star!

🐛 Found a bug? [Open an issue](https://github.com/yourusername/catboost-iris-classifier/issues)