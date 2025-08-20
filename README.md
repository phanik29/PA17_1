### Overview

This project compares the performance of multiple machine learning classifiers—k-nearest neighbors, logistic regression, decision trees, and support vector machines—on a real-world bank marketing dataset. The dataset is sourced from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) and contains results from Portuguese banking institution marketing campaigns. The analysis includes thorough data cleaning, handling of missing and 'unknown' values, and exploration of both categorical and numerical features. The goal is to identify the most accurate and interpretable model for predicting client subscription, supported by visualizations and clear business insights.

### Project Structure
The project is structured as follows:
```
.
├── README.md: Project overview and summary of findings
├── prompt_III.ipynb : Jupyter notebook containing the analysis and model comparisons
├── CRISP-DM-BANK.pdf: CRISP-DM methodology document
├── data
│   ├── bank-additional.csv
│   └── bank-additional-names.txt
│   └── bank-additional-full.csv
```

### Summary of Findings
This project analyzes a bank marketing dataset with 41,188 records and 20 features from the UC Irvine Machine Learning Repository. The dataset contains 11 categorical and 10 numerical features with an imbalanced target variable. Key data cleaning steps included handling 'unknown' values and removing 12 duplicate rows. 

**Key Insights:**
- Call duration shows the strongest correlation with subscription likelihood
- Economic indicators (emp.var.rate, cons.price.idx, euribor3m) are strong predictors
- Higher education levels correlate with increased subscription rates
- Cellular contact method outperforms telephone communication
- Seasonal patterns exist in subscription rates across months and days
- Multiple campaign contacts show diminishing returns

Multiple classifiers (k-nearest neighbors, logistic regression, decision trees, and support vector machines) were compared using cross-validation and grid search hyperparameter tuning. The analysis provides actionable insights for improving bank marketing campaign effectiveness.

**Model Performance Comparison:**

| Model                    | Train Time (s) | Train Accuracy | Test Accuracy | Best Parameters (if tuned) |
|--------------------------|----------------|----------------|---------------|----------------------------|
| Logistic Regression (Base) | 9.52        | 85.89%         | 86.53%        | Default                    |
| Logistic Regression (Tuned)| 107.54      | 85.92%         | 86.54%        | C=100, penalty=l1, solver=saga |
| KNN (Base)               | 0.10           | 92.98%         | 90.35%        | Default                    |
| KNN (Tuned)              | 135.27         | 100.00%        | 90.33%        | n_neighbors=5, weights=distance, metric=euclidean |
| Decision Tree (Base)     | 0.34           | 100.00%        | 89.41%        | Default                    |
| Decision Tree (Tuned)    | 29.13          | 90.73%         | 91.30%        | criterion=entropy, max_depth=3 |
| SVM (Base)               | 14.26          | 92.18%         | 91.49%        | Default                    |
| SVM (Tuned)              | 378.92         | 90.31%         | 90.60%        | C=10, kernel=linear        |

**Key Findings:**
- Decision Tree (Tuned) achieved the highest test accuracy (91.30%) with reduced overfitting through depth constraints
- Hyperparameter tuning significantly improved Decision Tree generalization while reducing overfitting
- SVM showed slight performance decrease after tuning, suggesting base parameters were already well-suited
- Training times increased substantially with grid search, particularly for SVM (378.92s vs 14.26s)

### Future Observations

- Explore ensemble methods (e.g., Random Forest, Gradient Boosting) to potentially improve predictive performance.
- Investigate feature selection techniques to reduce dimensionality and enhance model interpretability.
- Address class imbalance using resampling or cost-sensitive learning to further improve minority class prediction.
- Evaluate model performance using additional metrics such as precision, recall, F1-score, and ROC-AUC.
- Monitor model drift and retrain periodically as new data becomes available.
