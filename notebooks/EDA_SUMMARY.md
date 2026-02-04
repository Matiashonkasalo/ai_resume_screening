##  Notebook Summary & Baseline Results

### What we did in this notebook
In this notebook, I performed an end-to-end exploratory analysis of the AI-Driven Resume Screening dataset. The workflow included:

- Loading and inspecting the dataset structure and schema  
- Analyzing the target variable (`shortlisted`) and identifying a moderate class imbalance (~70% Yes / 30% No)  
- Performing univariate and bivariate visual analysis to understand feature distributions and relationships  
- Examining correlations between numeric features and assessing multicollinearity using VIF  
- Building a **baseline Logistic Regression model** without applying class balancing or hyperparameter tuning  
- Evaluating model performance using standard classification metrics and a confusion matrix  

The goal was not to optimize performance, but to establish a **transparent, interpretable baseline** and validate that the dataset contains meaningful predictive signal.

---

### Baseline model performance

The baseline Logistic Regression model achieved the following results on the test set:

- **Accuracy:** 91%  
- **Precision (Yes):** 0.93  
- **Recall (Yes):** 0.94  
- **F1-score (Yes):** 0.93  
- **Precision (No):** 0.85  
- **Recall (No):** 0.84  


This performance significantly outperforms the naïve majority-class baseline (~70% accuracy), confirming that the dataset contains strong predictive signal.

---

### Interpretation
- The model performs particularly well at identifying shortlisted candidates, as reflected by the high recall and precision for the “Yes” class.  
- Performance on the “No” class remains reasonable despite class imbalance, though some bias toward predicting “Yes” is expected.  
- The results suggest that features related to experience, skills alignment, project exposure, and resume characteristics are informative for shortlisting decisions.  

Overall, this baseline establishes confidence in both the data quality and the modeling approach.

---

## Next Steps 

In subsequent stages of the project, the following improvements will be explored in **modular Python code outside the notebook**:

- Addressing class imbalance using class weighting or resampling techniques  
- Experimenting with more expressive models (e.g., tree-based and ensemble methods)  
- Feature engineering and feature selection informed by EDA findings  
- Hyperparameter tuning and systematic model comparison  
- Experiment tracking and reproducible evaluation workflows  
- Production-oriented training and inference pipelines  


