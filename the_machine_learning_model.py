from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
import pandas as pd

data = pd.read_csv("titanic.csv")

df = data.copy()
df["Age"] = df["Age"].fillna(df["Age"].median())

df["Age Group"] = pd.cut(
    df["Age"],
    bins=[0, 12, 18, 25, 100],
    labels=["Child", "Teen", "Young", "Adult"]
)

df["Pclass_Category"] = df["Pclass"].map({
    1:"rich",
    2:"middle",
    3:"poor"
})

df["Family_Size"] = df["Siblings/Spouses Aboard"] + df["Parents/Children Aboard"] + 1
df["Is_Alone"] = (df["Family_Size"] == 1).astype(int)
df["Fare_Group"] = pd.cut(df["Fare"], bins=4)

df = pd.get_dummies(df, drop_first=True)

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42
)

pipeline = Pipeline([
    ("scalar", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print("Accuracy", accuracy_score(y_test, y_pred))