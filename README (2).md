# Spam Email Detection using Machine Learning

A binary text classifier that labels emails as **spam** or **ham** (not spam). It converts email text into TF-IDF features and compares two models: **Naive Bayes** and **Logistic Regression**.

## Dataset

- Email spam dataset from Kaggle: Spam | Ham Email Dataset for Classification [https://www.kaggle.com/datasets/meruvulikith/190k-spam-ham-email-dataset-for-classification]
- About 193,850 emails after cleaning (2 rows with missing text removed, 0 duplicates)
- 91,691 spam and 102,159 ham (47.3% spam, nearly balanced)
- The dataset is **not included** in this repository because of its size. Download it and save it as `spam.csv` in the project folder.

## Method

1. Load the data and find the label and text columns
2. Data quality checks: missing values and duplicates
3. Class balance and message length exploration
4. Encode labels (spam = 1, ham = 0) and split 80/20 (stratified, `random_state=42`): 155,080 train and 38,770 test emails
5. TF-IDF features (lowercase, English stop words removed) inside a scikit-learn `Pipeline`, so the test data never influences the vocabulary
6. Train Naive Bayes and Logistic Regression
7. Evaluate with accuracy, precision, recall, F1 and a confusion matrix
8. Inspect the top spam-indicating words
9. Predict new messages and save the model with `joblib`

## Results (test set, 38,770 emails)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes | 96.60% | 98.61% | 94.14% | 96.32% |
| **Logistic Regression** | **97.83%** | 96.91% | **98.56%** | **97.73%** |

Confusion matrix for Logistic Regression:

| | Predicted ham | Predicted spam |
|---|---|---|
| Actually ham | 19,856 | 576 |
| Actually spam | 264 | 18,074 |


Top spam-indicating words included `http`, `php`, `money`, `save`, `remove`, `pills` and `offer`. A few high-weight terms (`escapenumbero`, `bescapenumber`) are artifacts of how the dataset was preprocessed, and `hk` likely reflects the data source, so they are not real spam signals.

## How to run

```
pip install -r requirements.txt
python spam_detection.py
```

Place `spam.csv` in the same folder first. The script finds the label and text columns by name, so most email spam CSVs work without changes. Charts open in separate windows; close each one to continue.

## Limitations

- Trained and tested on a single dataset, so results may not carry over to emails from other sources
- Uses only the words in the text (no sender, links or attachments) and ignores word order
- Dataset artifacts such as `escapenumber` are still present in the text

## Next steps

- Remove dataset artifacts and re-evaluate
- Add n-grams and tune settings with cross-validation
- Adjust the decision threshold to reduce false positives
- Test on emails from a different dataset

## Author

Kota Reddy Jeswitha, B.E. Computer Science and Engineering, R.M.K. College of Engineering and Technology
