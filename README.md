# Credit-Scoring-German-Credit-Risk-Dataset-

1) What the project is about
The goal of the project is to build a credit scoring model that uses customer data to predict whether they are a reliable or risky borrower.
This is a classic binary classification task.

What is important in business terms:
the mistake of “missing a risky customer” is usually more expensive than “rejecting a reliable one.” Therefore, we focus not only on Accuracy, but also use more adequate metrics (ROC-AUC and F1).

2) Data
The German Credit dataset is used.

Target variable
credit_risk:

1 — good (reliable)
0 — bad (risky)
The data contains approximately:

70% good
30% bad
That is, there is a moderate class imbalance. Therefore, Accuracy is not the main metric.
Feature types
We divided the features into:

numerical
categorical
3) EDA (Exploratory Data Analysis) — what we understood
The task of EDA is not “just to draw graphs,” but to understand patterns in order to then correctly perform preprocessing and select models.

Main conclusions of EDA
Numerical features:

credit_amount has a strong right-sided asymmetry (rare but very large loans).
The growth of credit_amount and duration_months is associated with an increase in credit risk.
Categorical features have a strong impact on risk:

the worst risk is among customers with a negative account balance
the best risk is among customers with “no checking account” and stable features
This explains why categorical features are important and why CatBoost is a good candidate.
Due to class imbalance:

we will look at ROC-AUC and and F1-score, not just accuracy.

4) Feature Engineering — what we added and why
Based on EDA, we added four new features (the same for train and test, without target leakage):

credit_load = credit_amount / duration_months
(monthly credit load)
is_young = 1 if age < 25
(young borrower)
has_savings = 1 if savings_account != “< 100 DM”
(has savings)
stable_job = 1 if employment_duration ∈ {“4–7 years”, “>= 7 years”}
(stable employment)
After adding features, the dimension became: (800, 24) on train. 

Why this is useful:
we “packaged” the meaning into numerical form — models catch patterns more easily.

5) Data Preprocessing — INDIVIDUALLY for each model
Key point of the project: we did NOT do one preprocessing for all, because different models “like” different data representations. 

Below is strictly “what we did” for each model.

5.1) Logistic Regression — preprocessing for a linear model
Why: Logistic Regression is sensitive to:

the scale of numerical features
multicollinearity (repeated information from categories)
What we did:

Numerical features → StandardScaler
Categorical features → OneHotEncoder(drop=“first”)
(drop=“first” reduces multicollinearity)
Class imbalance was taken into account during training using class_weight="balanced"
After conversion, we got:

X_train_lr.shape = (800, 52)
X_test_lr.shape  = (200, 52)

5.2) Random Forest — preprocessing for trees
Why: Random Forest (trees) is almost insensitive to scale, so scaling is usually not necessary.

What we did:

Numeric features → passthrough (without scaling)
Categorical features → OneHotEncoder(drop=None, handle_unknown="ignore") 
Final dimension:
train: (800, 65)
test: (200, 65)
5.3) CatBoost — minimal preprocessing (its superpower)
Why: CatBoost can work with categories “smartly” through its internal statistics, so One-Hot often even worsens the situation.

What we did:

Features are passed almost “as is” (numerical + categorical)
Categorical features are passed through cat_features indexes
Dimension:
train: (800, 24)
test: (200, 24)
Number of categorical features: 13 


6) Baseline Model — starting point
Logistic Regression was chosen as the baseline because:

it is simple
it is interpretable
it is a good “zero point” before more powerful models
Baseline parameters:

solver="liblinear"
class_weight=“balanced”
max_iter=3000
Baseline metrics (test):

ROC-AUC = 0.7589
F1-score = 0.7460
Threshold = 0.5

7) Hyperparameter Tuning (Optuna) — what we improved and how
We performed tuning using Optuna and made sure to:

divide train into train/validation so as not to use test when selecting
optimize not just one metric, but a combination of metrics.
7.1) What was the goal of optimization
Target function:
F1-score × ROC-AUC

The meaning is simple:

ROC-AUC answers: “how well does the model rank customers by risk”
F1 answers: “how well does the model catch class 1 at the selected threshold”
the product forces the model to be “both smart in probabilities and useful in 0/1 decisions.”

8) Threshold selection — what it is and why it is needed
Most models output the probability P(class=1).

To get a final answer of 0/1, you need a threshold:

if proba >= threshold → 1
otherwise → 0
Why we tuned the threshold:
the threshold of 0.5 is the “default,” but with class imbalance and the business cost of errors, it is almost always more profitable to select a threshold that improves F1 and error balance.

9) Model tuning — what exactly was selected
9.1) Logistic Regression (Tuned)
What was optimized:

C (regularization strength)
penalty (l1 or l2)
threshold
Best Optuna parameters:

C = 0.3110043457531729
penalty = “l1”
threshold = 0.30243874403649756
Metrics on test (final):

ROC-AUC = 0.7592
F1-score = 0.7817
Threshold = 0.30243874403649756
What improved: F1 increased compared to the baseline (0.7460 → 0.7817) because we moved away from the 0.5 threshold to a more suitable one.

9.2) Random Forest (Tuned)
What we optimized:

n_estimators
max_depth
min_samples_split
min_samples_leaf
max_features
threshold
Best Optuna parameters:

n_estimators = 566
max_depth = 6
min_samples_split = 13
min_samples_leaf = 2
max_features = “sqrt”
threshold = 0.40444391169605093
Metrics on test (final):

ROC-AUC = 0.7881
F1-score = 0.8333
Threshold = 0.40444391169605093

9.3) CatBoost (Tuned)
What we optimized:

iterations
depth
learning_rate
l2_leaf_reg
random_strength
threshold
Best Optuna parameters:

iterations = 858
depth = 4
learning_rate = 0.09041179707806742
l2_leaf_reg = 4.197379121804284
random_strength = 0.715638441058104
threshold = 0.4871332648045998
Metrics on test (final):

ROC-AUC = 0.7645
F1-score = 0.8151
Threshold = 0.4871332648045998

10) Final comparison of models (test)
Based on the results of the final comparison:

Logistic Regression (Tuned): ROC-AUC 0.7592, F1 0.7817, thr≈0.30
Random Forest (Tuned): ROC-AUC 0.7881, F1 0.8333, thr≈0.40
CatBoost (Tuned): ROC-AUC 0.7645, F1 0.8151, thr≈0.49 29
Winner: Random Forest, as it gave the best metrics on the test.

11) What was saved (reproducibility)
To make the project “mature,” we saved artifacts via joblib:

train/test data after FE
preprocessors for LR and RF
cat_feature_indices for CatBoost
trained models and their metrics
