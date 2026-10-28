"""Explainable FinTech framework for credit scoring, personalised lending and
trustworthy financial decision-making.

Modules
-------
config         YAML configuration loader
data           dataset loading, synthetic data generator, stratified splits
preprocessing  normalisation, mean imputation, one-hot encoding (Eq. 7–11)
model          attention + Bi-LSTM + residual credit-risk network (Eq. 12–39)
fairness       demographic parity / equalised odds metrics and losses (Eq. 48–52)
train          custom training loop with Adam and fairness regulariser (Eq. 59–65)
uncertainty    Monte-Carlo dropout posterior approximation (Eq. 40)
explain        SHAP, LIME, integrated gradients and attention scores (Eq. 43–47)
lending        risk score, interest rate, approval, PRS and dynamic updates (Eq. 41–42, 53–58)
evaluate       classification metrics (accuracy, F1, AUC-ROC, MCC, ...)
plots          figure generation for every figure in the paper
"""

__version__ = "1.0.0"
