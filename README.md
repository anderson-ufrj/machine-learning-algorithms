# 🤖 Machine Learning Algorithms Collection
*Where classical statistics meets modern AI wizardry*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![IFSULDEMINAS](https://img.shields.io/badge/Institution-IFSULDEMINAS-red.svg)](https://portal.muz.ifsuldeminas.edu.br/)

> *"In the garden of algorithms, every flower tells a different story of classification"* 🌸

## 📚 Overview

This repository contains a comprehensive collection of supervised machine learning algorithms implemented during the AI course at **IFSULDEMINAS – Campus Muzambinho**. Each algorithm is meticulously crafted and tested on the legendary **Iris dataset** – because sometimes the classics are classics for a reason.

Think of this as your digital herbarium of machine learning models, where each algorithm is a different lens through which we can understand the art of classification.

## 🔬 Algorithms Implemented

### 🎯 **Logistic Regression** (`regressao_logistica.ipynb`)
The elegant statistician of the bunch. Despite its name suggesting regression, it's actually a classification powerhouse that uses the sigmoid function to squeeze probabilities into neat little boxes.

**Key Features:**
- Binary and multiclass classification
- Probabilistic outputs
- Linear decision boundaries
- Interpretable coefficients

### 🎲 **Naive Bayes** (`Naive_Bayes.ipynb`)
The optimistic probabilist who assumes independence even when the world is beautifully interconnected. Sometimes naive assumptions lead to surprisingly sophisticated results.

**Key Features:**
- Bayesian probability framework
- Fast training and prediction
- Handles categorical and continuous features
- Excellent for text classification (though we're using flowers here)

### 🌳 **Random Forest** (`Random_Forest.ipynb`)
Democracy in action – where multiple decision trees vote to reach a consensus. Because wisdom of crowds often trumps individual genius.

**Key Features:**
- Ensemble of decision trees
- Built-in feature importance
- Handles overfitting gracefully
- Robust to outliers

### ⚡ **XGBoost** (`Treinado_com_XGBoost.ipynb`)
The speed demon with a PhD in gradient optimization. When you need results yesterday and accuracy tomorrow.

**Key Features:**
- Extreme gradient boosting
- High performance and scalability
- Advanced regularization
- Feature importance ranking

### 💡 **LightGBM** (`Gradient_Boosting_usando_LightGBM.ipynb`)
The minimalist's dream – maximum efficiency with minimum memory footprint. Light on resources, heavy on results.

**Key Features:**
- Leaf-wise tree growth
- Memory efficient
- GPU acceleration support
- Fast training speed

### 🐱 **CatBoost** (`Gradient_Boosting_CatBoost.ipynb`)
The categorical data whisperer. Handles mixed data types like a Swiss Army knife handles camping situations.

**Key Features:**
- Native categorical feature support
- Symmetric trees
- Ordered boosting
- Minimal hyperparameter tuning

## 🌸 Dataset: The Timeless Iris

The **Iris dataset** – Ronald Fisher's 1936 gift to machine learning that keeps on giving. Three species of iris flowers (*setosa*, *versicolor*, *virginica*) measured across four features:

- **Sepal Length** (cm)
- **Sepal Width** (cm)  
- **Petal Length** (cm)
- **Petal Width** (cm)

*150 samples, 3 classes, infinite possibilities.*

## 🚀 Getting Started

### Prerequisites
```bash
# The usual suspects
pip install numpy pandas scikit-learn matplotlib seaborn
pip install xgboost lightgbm catboost
pip install jupyter notebook
```

### Quick Start
```bash
# Clone this digital garden
git clone https://github.com/anderson-ufrj/machine-learning-algorithms.git
cd machine-learning-algorithms

# Fire up Jupyter
jupyter notebook

# Pick your algorithm and let the magic begin ✨
```

## 📊 Performance Metrics

Each notebook includes comprehensive evaluation metrics:

- **Accuracy Score** - The classic crowd-pleaser
- **Precision & Recall** - For when you need to be more nuanced
- **F1-Score** - The harmonic mean that keeps everyone happy
- **Confusion Matrix** - Visual truth in a grid
- **Classification Report** - The full statistical symphony
- **Cross-Validation** - Because one split is never enough

## 🎨 Visualizations

Every notebook comes with rich visualizations:
- Decision boundaries (where possible)
- Feature importance plots
- Learning curves
- Confusion matrices with heatmaps
- ROC curves for the probability enthusiasts

## 🏗️ Project Structure

```
├── 📁 notebooks/
│   ├── 📓 regressao_logistica.ipynb
│   ├── 📓 Naive_Bayes.ipynb
│   ├── 📓 Random_Forest.ipynb
│   ├── 📓 Treinado_com_XGBoost.ipynb
│   ├── 📓 Gradient_Boosting_usando_LightGBM.ipynb
│   └── 📓 Gradient_Boosting_CatBoost.ipynb
├── 📄 README.md
└── 📄 requirements.txt
```

## 🎯 Learning Objectives

This collection serves multiple pedagogical purposes:

1. **Algorithm Understanding** - Deep dive into how each method thinks
2. **Comparative Analysis** - See how different approaches tackle the same problem  
3. **Implementation Skills** - Hands-on coding with real libraries
4. **Evaluation Techniques** - Master the art of model assessment
5. **Visualization** - Tell stories with data and graphs

## 🔮 Future Enhancements

- [ ] Deep learning implementations (Neural Networks)
- [ ] Unsupervised learning algorithms
- [ ] Advanced ensemble methods
- [ ] Hyperparameter optimization tutorials
- [ ] Model deployment examples
- [ ] Performance benchmarking suite

## 📖 References & Inspiration

- Fisher, R.A. (1936). "The use of multiple measurements in taxonomic problems"
- Scikit-learn documentation
- XGBoost, LightGBM, and CatBoost official documentation
- IFSULDEMINAS AI Course Materials

## 🤝 Contributing

Found a bug? Have an idea? Want to add another algorithm to the collection? 

```bash
# Fork, code, commit, push, PR
# The eternal dance of open source collaboration
```

## 📜 License

MIT License – because knowledge should be free to roam and reproduce.

## 👨‍💻 Author

**Anderson Silva** - *Digital Intelligence Architect*
- 🔗 LinkedIn: [anderson-h-silva95](https://www.linkedin.com/in/anderson-h-silva95/)
- 🐦 Twitter: [@neural_thinker](https://twitter.com/neural_thinker)
- 📧 Email: andersonhs27@gmail.com
- 🐙 GitHub: [anderson-ufrj](https://github.com/anderson-ufrj)

## 🎵 *Coda*

*"In the symphony of machine learning, each algorithm plays its own instrument. Some are violins – precise and melodic. Others are drums – powerful and rhythmic. Together, they create the music of artificial intelligence."*

---

**Tags:** `#MachineLearning` `#AI` `#Python` `#Scikit-learn` `#DataScience` `#Classification` `#Algorithms` `#IFSULDEMINAS` `#Iris` `#XGBoost` `#LightGBM` `#CatBoost` `#RandomForest` `#NaiveBayes` `#LogisticRegression`

*Made with ❤️ and countless cups of coffee in Muzambinho, MG* ☕
