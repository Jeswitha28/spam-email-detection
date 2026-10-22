# Spam Email / SMS Detection using Machine Learning
# ---------------------------------------------------
# Run in VS Code: each "# %%" block is a cell (click "Run Cell"), or run the
# whole file with:  python spam_detection.py
#
# Install once:
#   pip install pandas scikit-learn matplotlib seaborn joblib
#
# Dataset: the "SMS Spam Collection" (spam.csv) from Kaggle / UCI.
# Put spam.csv in the same folder as this file.

# %% 1. Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

DATA_PATH = "spam.csv"

# %% 2. Load the data
# Works with email datasets (columns like label/text) and the SMS dataset
# (columns v1/v2). We find the label and text columns by name, keep only
# those two, and make sure the label is the words "spam" / "ham".
df = pd.read_csv(DATA_PATH, encoding="latin-1")
cols = {c.lower().strip(): c for c in df.columns}

label_col = next(cols[c] for c in ["label", "category", "v1", "class", "target"] if c in cols)
text_col = next(cols[c] for c in ["text", "message", "v2", "email", "body", "content"] if c in cols)

df = df[[label_col, text_col]]
df.columns = ["label", "text"]

# If labels are numbers (1 = spam, 0 = ham), convert them to words
if pd.api.types.is_numeric_dtype(df["label"]):
    df["label"] = df["label"].map({1: "spam", 0: "ham"})

print("Shape:", df.shape)
print(df.head())

# %% 3. Data quality checks
print("\nMissing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

df = df.dropna().drop_duplicates().reset_index(drop=True)
df["label"] = df["label"].str.strip().str.lower()
print("Shape after cleaning:", df.shape)
df["text"] = df["text"].str.replace("escapenumber", " ", regex=False)
# %% 4. Class balance (important: spam data is usually imbalanced)
counts = df["label"].value_counts()
print("\nClass counts:\n", counts)
print("Spam share: {:.1f}%".format(100 * counts.get("spam", 0) / len(df)))

sns.countplot(data=df, x="label")
plt.title("Spam vs Ham")
plt.show()

# %% 5. Simple feature exploration: spam messages tend to be longer
df["length"] = df["text"].str.len()
print(df.groupby("label")["length"].mean().round(1))

sns.histplot(data=df, x="length", hue="label", bins=40)
plt.title("Message length by class")
plt.show()

# %% 6. Encode labels and split into train / test
# spam = 1, ham = 0.  stratify keeps the same spam share in both sets.
df["target"] = (df["label"] == "spam").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["target"],
    test_size=0.2, random_state=42, stratify=df["target"],
)
print("Train size:", len(X_train), "| Test size:", len(X_test))

# %% 7. Build and train two models
# TF-IDF turns each message into numbers. It lowercases, removes English stop
# words, and gives higher weight to words that are frequent in a message but
# rare overall.  Using a Pipeline means the same preprocessing is applied at
# training time and at prediction time.
def make_pipeline(model):
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
        ("model", model),
    ])

models = {
    "Naive Bayes": make_pipeline(MultinomialNB()),
    "Logistic Regression": make_pipeline(
        LogisticRegression(max_iter=1000, class_weight="balanced")
    ),
}

for name, pipe in models.items():
    pipe.fit(X_train, y_train)

# %% 8. Evaluate: accuracy alone is NOT enough on imbalanced data
rows = []
for name, pipe in models.items():
    pred = pipe.predict(X_test)
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
    })
results = pd.DataFrame(rows).set_index("Model").round(4)
print("\n", results)

# %% 9. Confusion matrix for the better model (by F1)
best_name = results["F1"].idxmax()
best_model = models[best_name]
pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, pred)
print("\nBest model:", best_name)
print(classification_report(y_test, pred, target_names=["ham", "spam"], zero_division=0))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - " + best_name)
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"True ham kept: {tn} | Ham wrongly flagged as spam (false positives): {fp}")
print(f"Spam missed (false negatives): {fn} | Spam caught: {tp}")

# %% 10. Which words push a message towards spam? (Logistic Regression)
lr_pipe = models["Logistic Regression"]
words = lr_pipe.named_steps["tfidf"].get_feature_names_out()
coefs = lr_pipe.named_steps["model"].coef_[0]
top = pd.Series(coefs, index=words).sort_values(ascending=False).head(15)
print("\nTop spam-indicating words:\n", top.round(2))

top.sort_values().plot(kind="barh", figsize=(7, 5))
plt.title("Top 15 spam words (Logistic Regression weights)")
plt.show()

# %% 11. Predict on new messages
def predict_message(text, model=best_model):
    label = model.predict([text])[0]
    return "SPAM" if label == 1 else "HAM"

samples = [
    "Congratulations! You won a free prize. Call now to claim your reward",
    "Hey, are we still meeting for lunch tomorrow?",
    "URGENT! Your account is locked. Click the link to verify now",
]
for s in samples:
    print(predict_message(s), "->", s)

# %% 12. Save the trained model so it can be reused without retraining
joblib.dump(best_model, "spam_model.joblib")
print("\nSaved spam_model.joblib")
# To load later:  model = joblib.load("spam_model.joblib")
