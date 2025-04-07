
# 🎯 LassoLens

**Spotlight the Signal. Shrink the Noise.**

**LassoLens** is a hands-on Python project that dives deep into multiple linear regression with a focus on **regularization** techniques—specifically **Lasso**, **Ridge**, and **Ordinary Least Squares**. It demonstrates how **Lasso Regression** can be a powerful tool for **feature selection** in high-dimensional datasets.

---

## 📌 Project Highlights

- ✅ Generate synthetic regression data with high dimensions and sparse signal using `make_regression`
- 🧪 Compare performance of **Linear Regression**, **Ridge**, and **Lasso**
- 🎯 Use **Lasso** to select informative features by inspecting residual coefficient magnitudes
- 📉 Visualize coefficient weights, prediction accuracy, and model performance before and after feature selection
- 📊 Print detailed metrics: R², MSE, MAE, and Explained Variance

---

## 🚀 Quick Start

1. **Clone the repo**
   ```bash
   git clone https://github.com/Kirankumarvel/LassoLens.git
   cd LassoLens
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**
   Open `lasso_feature_selection.ipynb` in Jupyter or VS Code and run all cells.

---

## 📁 File Structure

```
LassoLens/
├── lassolens.py     # Main project notebook
├── requirements.txt                  # Dependencies
└── README.md                         # You’re here
```

---

## 🧠 Concepts Covered

- Multiple Linear Regression
- Regularization: Ridge vs Lasso
- Coefficient Analysis & Thresholding
- Feature Selection & Model Re-training
- Visual Diagnostics (prediction vs actual, coefficient plots)

---

## 📊 Example Plots

- Coefficient comparison for Linear, Ridge, and Lasso
- Predicted vs Actual values for each model
- Line plots showing prediction alignment
- Filtered feature importance table

---

## 🧪 Requirements

- Python 3.7+
- `scikit-learn`
- `matplotlib`
- `numpy`
- `pandas`

(Install all with `pip install -r requirements.txt`)

---

## 📬 Feedback or Ideas?

Feel free to open an [issue](https://github.com/Kirankumarvel/LassoLens/issues) or drop a pull request! Let’s make this cleaner and sharper together.

---

## ⭐️ Support

If you like this project, consider giving it a ⭐️! It helps others discover it.

---

> *“In a world full of noise, be a lens that finds the signal.” – LassoLens*
