# Setup
ML-Practice/
    01_Python_NumPy/
        numpy_basics.ipynb
        python_tricks.ipynb
        practice_exercises.ipynb

    02_Pandas_Data_Cleaning/
        data_inspection.ipynb
        missing_values.ipynb
        encoding_scaling.ipynb
        groupby_merge.ipynb

    03_Machine_Learning/
        linear_regression.ipynb
        logistic_regression.ipynb
        knn.ipynb
        svm.ipynb
        decision_trees.ipynb
        random_forest.ipynb
        clustering_pca.ipynb

    04_Deep_Learning/
        pytorch_basics.ipynb
        training_loop.ipynb
        multimodal_learning.ipynb

    05_Assessment_Practice/
        question_01.ipynb
        question_02.ipynb
        mock_test_01.ipynb

    datasets/
    requirements.txt
    README.md


# Practical Python and Machine Learning Coding Preparation

This guide collects the practical Python, data-cleaning, model-training,
evaluation, and multimodal ML knowledge discussed in the conversation.
It is designed for data-scientist coding assessments where starter code
is provided and you must fill in missing sections.

------------------------------------------------------------------------

## 1. What you need to practise

Understanding ML concepts and implementing them in Python are separate
skills. Build comfort in three areas:

1.  **Python for data manipulation:** loading datasets, cleaning data,
    handling missing values, encoding categorical variables, and
    inspecting data.
2.  **Python for machine learning:** splitting datasets, training
    models, predicting, evaluating, and tuning hyperparameters.
3.  **Practical coding:** reading partially completed code and
    implementing missing pieces without needing to remember every
    library function.

------------------------------------------------------------------------

# 2. Python tools for discovering syntax

## A. Inspecting functions and objects

  --------------------------------------------------------------------------------
  Function / feature      Purpose                 Example
  ----------------------- ----------------------- --------------------------------
  `help()`                Read documentation      `help(pd.DataFrame.fillna)`

  `dir()`                 List available          `dir(df)`
                          attributes and methods  

  `type()`                Find an object's type   `type(X_train)`

  `isinstance()`          Check an object's type  `isinstance(df, pd.DataFrame)`

  `len()`                 Get number of elements  `len(df)`

  `print()`               Inspect values          `print(df.head())`

  `vars()`                Inspect an object's     `vars(model)`
                          attribute dictionary,   
                          when available          

  `inspect.signature()`   See function parameters `inspect.signature(model.fit)`
  --------------------------------------------------------------------------------

Example:

``` python
import inspect
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()

help(model.fit)
print(inspect.signature(model.fit))
print(dir(model))
```

Use these tools when you remember a function or object name but forget
its syntax, arguments, or available methods.

## B. Jupyter Notebook and Google Colab shortcuts

  Shortcut             Purpose
  -------------------- -------------------------------------------
  `function?`          Show documentation
  `function??`         Show source code, if available
  `Tab`                Autocomplete methods and variables
  `Shift + Tab`        Show function signature and documentation
  `Ctrl + Shift + -`   Split a cell at the cursor

For example, type `model.fit(` and press `Shift + Tab` to inspect the
method's parameters.

------------------------------------------------------------------------

# 3. Inspecting a dataset with Pandas

Start by loading a dataset:

``` python
import pandas as pd

df = pd.read_csv("data.csv")
```

Useful inspection commands:

  -----------------------------------------------------------------------
  Command                             What it tells you
  ----------------------------------- -----------------------------------
  `df.head()`                         First 5 rows

  `df.tail()`                         Last 5 rows

  `df.shape`                          Number of rows and columns

  `df.columns`                        Column names

  `df.info()`                         Data types and non-null counts

  `df.describe()`                     Statistical summary of numerical
                                      columns by default

  `df.dtypes`                         Data types of columns

  `df.isnull().sum()`                 Missing values per column

  `df.nunique()`                      Number of unique values per column

  `df.value_counts()`                 Frequency of values in a Series

  `df.duplicated().sum()`             Number of duplicate rows

  `df.corr(numeric_only=True)`        Correlations between numerical
                                      columns
  -----------------------------------------------------------------------

Whenever you receive an unfamiliar dataset, make this a reflex:

``` python
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
```

------------------------------------------------------------------------

# 4. Data cleaning: common operations

## A. Selecting columns and rows

``` python
# Select one column
df["age"]

# Select multiple columns
df[["age", "salary"]]

# Select rows based on a condition
df[df["age"] > 25]

# Multiple conditions: AND
df[(df["age"] > 25) & (df["salary"] > 50000)]

# Multiple conditions: OR
df[(df["age"] > 25) | (df["salary"] > 50000)]

# Drop a column
df = df.drop(columns=["id"])

# Drop rows by index
df = df.drop(index=[0, 1])
```

Remember: - `&` means AND. - `|` means OR. - Put parentheses around each
individual condition when combining Pandas conditions.

## B. Handling missing values

``` python
# Check missing values
df.isnull().sum()

# Remove rows containing missing values
df.dropna(inplace=True)

# Fill a numerical column with its mean
df["age"] = df["age"].fillna(df["age"].mean())

# Fill a numerical column with its median
df["salary"] = df["salary"].fillna(df["salary"].median())

# Fill a categorical column with its mode
df["city"] = df["city"].fillna(df["city"].mode()[0])
```

Common starting strategies:

  -----------------------------------------------------------------------
  Data type / situation               Common strategy
  ----------------------------------- -----------------------------------
  Numerical, approximately symmetric  Mean
  distribution                        

  Numerical, skewed distribution or   Median
  outliers                            

  Categorical                         Mode

  Time series                         Forward fill, backward fill, or a
                                      domain-specific method
  -----------------------------------------------------------------------

**Important: avoid data leakage.** Do not calculate imputation values
using the entire dataset before train-test splitting. Fit the imputer on
training data only, then apply it to the test data.

## C. Encoding categorical variables

Many ML algorithms require numerical inputs.

Example data:

``` python
df["gender"] = ["Male", "Female", "Male"]
```

Label encoding:

``` python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["gender"] = le.fit_transform(df["gender"])
```

One-hot encoding with Pandas:

``` python
df = pd.get_dummies(df, columns=["gender"])
```

One-hot encoding with scikit-learn:

``` python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown="ignore")
```

General rules: - **Nominal categories** (such as city or colour):
one-hot encoding is usually appropriate. - **Ordinal categories** (such
as low, medium, high): ordinal encoding can preserve the meaningful
order. - Avoid blindly applying label encoding to nominal features
because it can introduce an artificial ordering.

For a real ML workflow, fit encoders on training data only. A
scikit-learn `Pipeline` or `ColumnTransformer` can help enforce this.

## D. Feature scaling

Two common approaches:

``` python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Min-max scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

  -----------------------------------------------------------------------
  StandardScaler                      MinMaxScaler
  ----------------------------------- -----------------------------------
  Centres features around mean 0      Usually scales features to \[0, 1\]

  Scales to standard deviation 1      Uses feature minimum and maximum

  Does not constrain values to a      Sensitive to extreme
  fixed range                         minimum/maximum values
  -----------------------------------------------------------------------

Algorithms that often benefit from scaling: - KNN - SVM - Logistic
Regression - Linear Regression with regularization - Neural Networks -
K-Means - PCA

Decision Trees and Random Forests generally do not require feature
scaling.

**Important:** In a train-test workflow, call `fit_transform()` on
training features and only `transform()` on test features. Never fit the
scaler on the test set.

------------------------------------------------------------------------

# 5. The standard machine-learning workflow

Whenever you are asked to train a model, follow this sequence:

1.  Load the dataset.
2.  Explore the dataset (EDA).
3.  Separate features and target.
4.  Split into training and testing data.
5.  Preprocess the data.
6.  Initialize the model.
7.  Train the model.
8.  Make predictions.
9.  Evaluate performance.

## A. Load and inspect data

``` python
import pandas as pd

df = pd.read_csv("data.csv")

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
```

## B. Separate features and target

``` python
X = df.drop(columns=["target"])
y = df["target"]
```

-   `X` contains the input features.
-   `y` contains the target variable you want to predict.

## C. Train-test split

``` python
from sklearn.model_selection import train_test_split

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

Remember: - `X` contains input features. - `y` contains the target. -
`test_size=0.2` reserves 20% for testing. - `random_state=42` makes the
split reproducible.

For classification, especially with imbalanced classes, consider
stratification:

``` python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

`stratify=y` preserves approximately the same class proportions in the
training and testing sets.

## D. The common scikit-learn model pattern

Most scikit-learn estimators use the same basic interface:

``` python
model = ModelName(parameters)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

The model class and its parameters change, but the `fit()` and
`predict()` pattern is generally the same.

## E. Common models and imports

  -------------------------------------------------------------------------------------------
  Algorithm                           Import
  ----------------------------------- -------------------------------------------------------
  Linear Regression                   `from sklearn.linear_model import LinearRegression`

  Logistic Regression                 `from sklearn.linear_model import LogisticRegression`

  KNN Classifier                      `from sklearn.neighbors import KNeighborsClassifier`

  KNN Regressor                       `from sklearn.neighbors import KNeighborsRegressor`

  Decision Tree Classifier            `from sklearn.tree import DecisionTreeClassifier`

  Random Forest Classifier            `from sklearn.ensemble import RandomForestClassifier`

  SVM classifier                      `from sklearn.svm import SVC`

  Naive Bayes                         `from sklearn.naive_bayes import GaussianNB`

  K-Means                             `from sklearn.cluster import KMeans`

  PCA                                 `from sklearn.decomposition import PCA`
  -------------------------------------------------------------------------------------------

### KNN classification

``` python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Linear regression

``` python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Random Forest classification

``` python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

Notice how little the training and prediction code changes between
models.

------------------------------------------------------------------------

# 6. Model evaluation

Choose metrics based on the task.

## A. Classification metrics

``` python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

For multiclass classification, be aware that precision, recall, and F1
may require an `average` setting such as `"macro"`, `"micro"`, or
`"weighted"`.

## B. Regression metrics

``` python
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
```

  Classification     Regression
  ------------------ -------------------
  Accuracy           MSE
  Precision          MAE
  Recall             RMSE
  F1-score           R²
  Confusion matrix   Residual analysis

RMSE can be calculated as:

``` python
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
```

Do not blindly maximize accuracy. On imbalanced data, accuracy can be
misleading; inspect precision, recall, F1, and the confusion matrix.

------------------------------------------------------------------------

# 7. Scikit-learn Pipelines

A `Pipeline` combines preprocessing and model training. It also helps
prevent data leakage because each preprocessing step is fitted on the
training data.

Example with imputation, scaling, and KNN:

``` python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier(n_neighbors=5))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

What happens? 1. The imputer learns replacement values from training
data. 2. The scaler learns the training-data mean and standard
deviation. 3. KNN trains on the transformed training data. 4. During
prediction, the same learned preprocessing is applied to test data.

For datasets with both numerical and categorical columns,
`ColumnTransformer` can be used to apply different preprocessing to
different columns.

**Golden rule:** Fit your scaler, imputer, encoder, and feature selector
on training data only. Apply the learned transformations to validation
and test data.

------------------------------------------------------------------------

# 8. Predictions and probabilities

``` python
# Predicted classes
y_pred = model.predict(X_test)

# Predicted probabilities, when supported by the estimator
y_prob = model.predict_proba(X_test)

# Probability of the positive class in binary classification
y_prob_positive = model.predict_proba(X_test)[:, 1]
```

`predict_proba()` is available only for estimators that support
probability estimation. Some classifiers require configuration or a
different method to obtain probabilities.

------------------------------------------------------------------------

# 9. Cross-validation and hyperparameter tuning

## A. Cross-validation

``` python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

print(scores)
print(scores.mean())
```

This evaluates the estimator across five folds. Cross-validation is
useful for assessing how consistently a model performs across different
splits of the training data.

## B. GridSearchCV

Use `GridSearchCV` to search through combinations of hyperparameters:

``` python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

param_grid = {
    "n_neighbors": [3, 5, 7, 9],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}

grid = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)

best_model = grid.best_estimator_
```

Understand: - `param_grid`: Parameters and values to try. - `cv`: Number
of cross-validation folds. - `best_params_`: Best parameter combination
found. - `best_score_`: Best cross-validation score. -
`best_estimator_`: Estimator configured with the best parameters and
refitted on all supplied training data by default.

Keep the test set separate until final evaluation; do not use it to
choose hyperparameters.

------------------------------------------------------------------------

# 10. Multimodal machine learning

Multimodal ML uses two or more kinds of input data, such as: - Text -
Images - Audio - Video - Numerical or tabular data

Example: classify products using both product images and product
descriptions.

A simplified architecture:

``` text
Image input                    Text input
    |                              |
Image encoder                  Text encoder
(CNN / ViT)                    (BERT / Transformer)
    |                              |
Image features                 Text features
    \                              /
     \                            /
             Feature fusion
        (concatenation / attention)
                    |
             Classification head
             (linear layer + softmax)
                    |
               Predicted class
```

A simple multimodal workflow:

1.  Load each modality.
2.  Preprocess each modality separately.
3.  Pass each modality through its respective encoder.
4.  Extract feature representations.
5.  Combine representations (fusion).
6.  Pass combined features to a prediction head.
7.  Calculate loss.
8.  Perform backpropagation and optimizer updates.

## A. Feature fusion with PyTorch

``` python
import torch

# Image and text feature vectors
image_features = image_encoder(images)
text_features = text_encoder(input_ids)

# Combine features along the feature dimension
combined = torch.cat(
    [image_features, text_features],
    dim=1
)

# Classification
outputs = classifier(combined)
```

A common issue is tensor dimension mismatch. Inspect shapes:

``` python
print(image_features.shape)
print(text_features.shape)
print(combined.shape)
```

For example: - Image features: `(32, 512)` - Text features:
`(32, 768)` - Concatenated features along dimension 1: `(32, 1280)`

The batch dimensions must match for concatenation along the feature
dimension.

------------------------------------------------------------------------

# 11. PyTorch training loop

If an assignment involves neural networks or multimodal models, learn
this training pattern.

``` python
import torch
import torch.nn as nn
import torch.optim as optim

model = MyModel()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):

    model.train()

    for inputs, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

    print("Epoch:", epoch + 1, "Loss:", loss.item())
```

Purpose of each operation:

  -----------------------------------------------------------------------
  Operation                           Purpose
  ----------------------------------- -----------------------------------
  `model.train()`                     Enable training mode

  `optimizer.zero_grad()`             Clear gradients from the previous
                                      iteration

  `model(inputs)`                     Forward pass

  `criterion(outputs, labels)`        Calculate loss

  `loss.backward()`                   Backpropagation

  `optimizer.step()`                  Update model weights
  -----------------------------------------------------------------------

## A. Evaluation mode

``` python
model.eval()

with torch.no_grad():
    outputs = model(inputs)
```

-   `model.eval()` changes the behaviour of layers such as Dropout and
    BatchNorm.
-   `torch.no_grad()` disables gradient tracking, reducing memory use
    during inference/evaluation.

**Important loss-function distinction:** - `nn.CrossEntropyLoss` expects
raw logits, not softmax probabilities. For standard multiclass
classification, labels are usually class indices of type `torch.long`. -
`nn.BCEWithLogitsLoss` is commonly used for binary or multilabel
classification and expects raw logits. - For regression, common choices
include `nn.MSELoss` and `nn.L1Loss`.

For multimodal models, the model's `forward()` method may accept
multiple inputs, for example `outputs = model(images, input_ids)`. The
dataloader and training loop must pass the inputs in the format expected
by that model.

------------------------------------------------------------------------

# 12. General rules and practical tricks

  -----------------------------------------------------------------------
  Situation                           What to think of
  ----------------------------------- -----------------------------------
  Dataset provided                    `head()`, `info()`, `describe()`

  Missing values                      `isnull()`, `SimpleImputer`

  Categorical columns                 `OneHotEncoder`

  Different feature scales            `StandardScaler`

  Need training and testing           `train_test_split`

  Classification                      Classifier + classification metrics

  Predict numerical values            Regressor + regression metrics

  Find optimal parameters             `GridSearchCV`

  Evaluate stability                  Cross-validation

  Reduce dimensions                   PCA

  Cluster unlabeled data              K-Means

  Avoid overfitting                   Regularization, cross-validation,
                                      early stopping

  Deep learning                       PyTorch or TensorFlow

  Multiple input modalities           Separate encoders + fusion

  Reproducible results                Set random seeds / `random_state`
                                      where supported
  -----------------------------------------------------------------------

Additional rules:

### Rule 1: Understand the target variable first

-   Continuous numerical target → Regression.
-   Categorical target → Classification.
-   No target variable and the goal is to find groups → Clustering.

### Rule 2: Establish a baseline

Start with a simple model before trying complex models. Examples: -
Logistic Regression for classification. - Linear Regression for
regression.

### Rule 3: Watch out for data leakage

Never use information from the test set to train the model, fit
preprocessing steps, select features, or tune hyperparameters.

### Rule 4: Check shapes when debugging

``` python
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
```

For neural networks, inspect tensor shapes at each stage of the forward
pass.

### Rule 5: Choose metrics carefully

For imbalanced classification, accuracy may hide poor performance on the
minority class. Inspect precision, recall, F1, and the confusion matrix.

------------------------------------------------------------------------

# 13. Libraries to prioritize

You do not need to learn every Python library. For data-scientist coding
assessments, prioritize:

  -----------------------------------------------------------------------
  Library                 Main use                Priority
  ----------------------- ----------------------- -----------------------
  NumPy                   Arrays and mathematical Must know
                          operations              

  Pandas                  Data cleaning and       Must know
                          manipulation            

  Matplotlib              Data visualization      Must know

  Seaborn                 Statistical             Must know
                          visualization           

  Scikit-learn            Classical ML algorithms Must know

  PyTorch                 Neural networks and     Must know
                          multimodal ML           

  Hugging Face            Pretrained language and Next step
  Transformers            vision models           

  OpenCV                  Image processing        Next step
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 14. Suggested 16-day practice roadmap

Assume approximately 2--3 hours of focused practice each day.

  -----------------------------------------------------------------------
  Stage                   Topics                  Suggested time
  ----------------------- ----------------------- -----------------------
  1\. Python and NumPy    Arrays, indexing,       2 days
  fundamentals            slicing, broadcasting,  
                          vectorization, lambda,  
                          map, filter, sorting    

  2\. Pandas and data     Missing values,         3 days
  cleaning                duplicates, groupby,    
                          merge, encoding,        
                          scaling, feature        
                          selection, EDA          

  3\. Classical ML        Linear regression,      4 days
  implementation          logistic regression,    
                          KNN, SVM, decision      
                          trees, random forests,  
                          K-Means, PCA,           
                          evaluation and tuning   

  4\. Deep learning and   PyTorch, datasets,      4 days
  multimodal training     dataloaders, training   
                          loops, loss functions,  
                          optimizers, image/text  
                          encoders and feature    
                          fusion                  

  5\. Practical           Incomplete code,        3 days
  assessment simulation   debugging, filling      
                          missing sections,       
                          interpreting outputs,   
                          and implementing models 
                          under time constraints  

  **Total**                                       **16 days**
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 15. How to practise assessment-style questions

Do not only read notes. Write code repeatedly.

A useful practice format is:

1.  Receive a dataset or a small example.
2.  Read starter code with missing sections marked `TODO`.
3.  Implement the missing code.
4.  Run the code and inspect the output.
5.  Debug errors and explain why they occurred.
6.  Repeat with less guidance and eventually under a time limit.

Recommended starting order:

1.  Pandas and data cleaning.
2.  NumPy fundamentals.
3.  Scikit-learn and model training.
4.  PyTorch and multimodal training.

For each exercise, aim to understand: - What the code is expected to
do. - What shape and type each variable should have. - Which library
function fits the task. - What the function returns. - How to verify
that the output is correct.

------------------------------------------------------------------------

# 16. Final assessment checklist

Before submitting an ML coding assignment, check the following:

## Dataset and preprocessing

-   [ ] Did I inspect the dataset with `head()`, `info()`, and
    `describe()`?
-   [ ] Did I identify the target variable correctly?
-   [ ] Did I check missing values and duplicates?
-   [ ] Did I encode categorical variables appropriately?
-   [ ] Did I scale features if the algorithm needs it?
-   [ ] Did I fit preprocessing only on training data?

## Model training

-   [ ] Did I choose classification, regression, or clustering
    correctly?
-   [ ] Did I split the data appropriately?
-   [ ] Did I initialize the correct model?
-   [ ] Did I call `fit()` with the right inputs?
-   [ ] Did I predict using the correct test features?

## Evaluation

-   [ ] Did I use metrics suitable for the task?
-   [ ] Did I inspect the confusion matrix or regression errors where
    appropriate?
-   [ ] Did I avoid tuning against the test set?

## Deep learning / multimodal

-   [ ] Did I check input and output tensor shapes?
-   [ ] Did I use the correct loss function?
-   [ ] Did I clear gradients before backpropagation?
-   [ ] Did I call `loss.backward()` and `optimizer.step()`?
-   [ ] Did I use `model.eval()` and `torch.no_grad()` for evaluation?

------------------------------------------------------------------------

## The most important advice

Do not try to memorize 100 library functions. Learn common coding
patterns, know how to discover unfamiliar syntax using `help()`,
`dir()`, and autocomplete, and practise implementing the complete
workflow without looking at solutions.

The goal is to become comfortable translating an ML problem into working
Python code, especially when an assessment gives you partially completed
code and asks you to fill in the missing sections.
