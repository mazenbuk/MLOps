import pandas as pd

drug_df = pd.read_csv("Data/drug200.csv")
drug_df = drug_df.sample(frac=1)
drug_df.head(3)

from sklearn.model_selection import train_test_split

X = drug_df.drop("Drug", axis=1).values
y = drug_df.Drug.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=125
)

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

cat_col = [1, 2, 3]
num_col = [0, 4]

transform = ColumnTransformer(
    [
        ("encoder", OrdinalEncoder(), cat_col),
        ("num_imputer", SimpleImputer(strategy="median"), num_col),
        ("num_scaler", StandardScaler(), num_col),
    ]
)

pipe = Pipeline(
    steps=[
        ("preprocessor", transform),
        ("model", RandomForestClassifier(n_estimators=100, random_state=125)),
    ]
)

pipe.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, f1_score

predictions = pipe.predict(X_test)
accuacy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")

print("Accuracy:", str(round(accuacy, 2) * 100) + "%", "F1:", {round(f1, 2)})

with open("Results/metric.txt", "w") as outfile:
    outfile.write(f"\nAccuracy =  {round(accuacy, 2)}, F1 Score = {round(f1, 2)}.")


import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot()
plt.savefig("Results/model_results.png", dpi=120)

import skops.io as sio
from skops.io import dump, load, get_untrusted_types

sio.dump(pipe, "Model/drug_pipeline.skops")
unknown_types = get_untrusted_types(file="Model/drug_pipeline.skops")
print(unknown_types)

unknown_types = get_untrusted_types(file="Model/drug_pipeline.skops")
sio.load("Model/drug_pipeline.skops", trusted=unknown_types)
