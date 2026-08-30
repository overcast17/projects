# Feature Engineering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tested, leakage-safe transformer that derives the agreed Spaceship Titanic features and expose an explanatory feature-engineering block in the research notebook.

**Architecture:** Put reusable transformation logic in `src/features.py` as a scikit-learn-compatible `FeatureEngineer`. Its `fit` learns only training-fold spending medians and group frequencies; its `transform` derives features, handles malformed cabins, and drops raw identifiers. `Research.ipynb` imports that transformer for descriptive inspection only; all later model pipelines will instantiate a fresh transformer inside their CV pipeline.

**Tech Stack:** Python, pandas, NumPy, scikit-learn, pytest, Jupyter Notebook.

---

### Task 1: Define the feature-engineering contract with failing tests

**Files:**
- Create: `G:/projects/Spaceship_Titanic/tests/test_features.py`
- Create: `G:/projects/Spaceship_Titanic/src/__init__.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pandas as pd

from src.features import FeatureEngineer


def make_frame():
    return pd.DataFrame({
        "PassengerId": ["0001_01", "0001_02", "0002_01", "0003_01"],
        "Cabin": ["B/0/P", "F/12/S", np.nan, "malformed"],
        "Name": ["A", "B", "C", "D"],
        "Age": [12.0, 13.0, np.nan, 25.0],
        "RoomService": [0.0, np.nan, 5.0, 1.0],
        "FoodCourt": [0.0, 2.0, 0.0, 0.0],
        "ShoppingMall": [0.0, 0.0, 0.0, 0.0],
        "Spa": [0.0, 0.0, 0.0, 0.0],
        "VRDeck": [0.0, 0.0, 0.0, 0.0],
    })


def test_feature_engineer_creates_expected_features_and_drops_identifiers():
    transformed = FeatureEngineer().fit_transform(make_frame())

    assert list(transformed.index) == [0, 1, 2, 3]
    assert {"IsChild", "TotalSpend", "HasSpending", "Deck", "CabinNum", "Side", "GroupSize", "IsAlone"}.issubset(transformed.columns)
    assert {"PassengerId", "Cabin", "Name"}.isdisjoint(transformed.columns)
    assert transformed.loc[0, "IsChild"] == 1
    assert transformed.loc[1, "IsChild"] == 0
    assert pd.isna(transformed.loc[2, "IsChild"])
    assert transformed.loc[0, "TotalSpend"] == 0.0
    assert transformed.loc[1, "TotalSpend"] == 2.0
    assert transformed.loc[0, "HasSpending"] == 0
    assert transformed.loc[1, "HasSpending"] == 1
    assert transformed.loc[0, "Deck"] == "B"
    assert transformed.loc[1, "CabinNum"] == 12.0
    assert transformed.loc[1, "Side"] == "S"
    assert pd.isna(transformed.loc[3, "Deck"])


def test_feature_engineer_uses_training_fold_only_for_group_size_and_spending_medians():
    frame = make_frame()
    transformer = FeatureEngineer().fit(frame.iloc[:3])
    transformed = transformer.transform(frame.iloc[[0, 3]])

    assert transformed.loc[0, "GroupSize"] == 2
    assert transformed.loc[3, "GroupSize"] == 1
    assert transformed.loc[0, "IsAlone"] == 0
    assert transformed.loc[3, "IsAlone"] == 1
    assert transformed.loc[0, "TotalSpend"] == 0.0
```

- [ ] **Step 2: Run the tests and verify the expected failure**

Run: `python -m pytest tests/test_features.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'src.features'`.

- [ ] **Step 3: Commit the red test**

```powershell
git add tests/test_features.py src/__init__.py
git commit -m "test: define feature engineering behavior"
```

### Task 2: Implement the leak-safe transformer

**Files:**
- Create: `G:/projects/Spaceship_Titanic/src/features.py`
- Test: `G:/projects/Spaceship_Titanic/tests/test_features.py`

- [ ] **Step 1: Implement the minimal transformer**

```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    spending_columns = ("RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck")
    required_columns = ("PassengerId", "Cabin", "Name", "Age", *spending_columns)

    def fit(self, X, y=None):
        self._validate_columns(X)
        groups = self._group_id(X["PassengerId"])
        self.group_sizes_ = groups.value_counts(dropna=False)
        self.spend_medians_ = X.loc[:, self.spending_columns].median()
        return self

    def transform(self, X):
        self._validate_columns(X)
        out = X.copy()
        spending = out.loc[:, self.spending_columns].fillna(self.spend_medians_)
        out.loc[:, self.spending_columns] = spending
        out["TotalSpend"] = spending.sum(axis=1)
        out["HasSpending"] = (out["TotalSpend"] > 0).astype("int8")
        out["IsChild"] = np.where(out["Age"].notna(), (out["Age"] < 13).astype("int8"), np.nan)
        cabin = out["Cabin"].astype("string").str.extract(
            r"^(?P<Deck>[^/]+)/(?P<CabinNum>\d+)/(?P<Side>[PS])$"
        )
        out["Deck"] = cabin["Deck"]
        out["CabinNum"] = pd.to_numeric(cabin["CabinNum"], errors="coerce")
        out["Side"] = cabin["Side"]
        group_id = self._group_id(out["PassengerId"])
        out["GroupSize"] = group_id.map(self.group_sizes_).fillna(1).astype("int16")
        out["IsAlone"] = (out["GroupSize"] == 1).astype("int8")
        return out.drop(columns=["PassengerId", "Cabin", "Name"])

    @staticmethod
    def _group_id(passenger_id):
        return passenger_id.astype("string").str.split("_", n=1).str[0]

    def _validate_columns(self, X):
        missing = set(self.required_columns) - set(X.columns)
        if missing:
            raise ValueError(f"FeatureEngineer requires columns: {sorted(missing)}")
```

- [ ] **Step 2: Run the feature tests and verify they pass**

Run: `python -m pytest tests/test_features.py -v`

Expected: `2 passed`.

- [ ] **Step 3: Run static notebook-independent checks**

Run: `python -c "import pandas as pd; from src.features import FeatureEngineer; x=pd.read_csv('data/train.csv').drop(columns='Transported'); z=FeatureEngineer().fit_transform(x); assert len(x)==len(z); assert z.isna().sum().ge(0).all(); print(z.shape)"`

Expected: `(8693, 18)` and exit code 0.

- [ ] **Step 4: Commit the implementation**

```powershell
git add src/features.py tests/test_features.py
git commit -m "feat: add leak-safe feature engineering"
```

### Task 3: Add the notebook feature-engineering section

**Files:**
- Modify: `G:/projects/Spaceship_Titanic/Research.ipynb`
- Test: `G:/projects/Spaceship_Titanic/tests/test_features.py`

- [ ] **Step 1: Add a Markdown section after EDA**

Add a heading `# 4. Feature Engineering` followed by Russian prose that defines all eight derived features, says that `GroupId` is internal only, and explicitly states that future models must create `FeatureEngineer()` inside their own pipelines rather than reuse preview data.

- [ ] **Step 2: Add a preview cell that does not train a model**

```python
from src.features import FeatureEngineer

X_raw = train.drop(columns="Transported")
y = train["Transported"].astype(int)

feature_preview = FeatureEngineer().fit_transform(X_raw)
print("Исходная форма:", X_raw.shape)
print("После feature engineering:", feature_preview.shape)
feature_preview.head()
```

- [ ] **Step 3: Add notebook assertions and descriptive output**

```python
expected_features = {
    "IsChild", "TotalSpend", "HasSpending", "Deck",
    "CabinNum", "Side", "GroupSize", "IsAlone",
}
assert len(feature_preview) == len(train)
assert expected_features.issubset(feature_preview.columns)
assert {"PassengerId", "Cabin", "Name"}.isdisjoint(feature_preview.columns)

display(feature_preview[list(expected_features)].head())
```

- [ ] **Step 4: Execute the notebook from the project root and confirm outputs**

Run: `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=180 Research.ipynb`

Expected: exit code 0; the new cells show source shape `(8693, 13)`, engineered shape `(8693, 18)`, and no assertion errors.

- [ ] **Step 5: Re-run tests and review the diff**

Run: `python -m pytest tests/test_features.py -v`

Expected: `2 passed`.

Run: `git diff --check -- Research.ipynb`

Expected: no output.

- [ ] **Step 6: Commit the notebook block**

```powershell
git add Research.ipynb
git commit -m "notebook: document feature engineering"
```
