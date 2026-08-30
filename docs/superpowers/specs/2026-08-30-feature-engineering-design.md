# Feature Engineering Design

## Goal

Add a leak-safe feature-engineering stage to `Research.ipynb` for the Spaceship Titanic classification project. The stage must be reusable by every classical-model cross-validation pipeline.

## Scope

The feature set will contain:

- `IsChild`: `Age < 13`; missing ages remain missing until numerical imputation.
- `TotalSpend`: row-wise sum of `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, and `VRDeck`.
- `HasSpending`: `TotalSpend > 0`; it remains a candidate feature and will later be tested by cross-validation together with `TotalSpend`.
- `Deck`, `CabinNum`, and `Side`: parsed from `Cabin` as `deck/cabin_number/side`.
- `GroupSize`: count of passengers with the same group portion of `PassengerId`.
- `IsAlone`: `GroupSize == 1`.

After extraction, the raw `Cabin`, `PassengerId`, `Name`, and temporary `GroupId` fields will not be passed to models. `GroupId` is used only internally to estimate group size; it will not be one-hot or target encoded.

## Data Flow and Leakage Control

`FeatureEngineer` will implement the scikit-learn transformer interface and sit as the first step of each model's `Pipeline`.

During `fit(X_train)`, it will derive group-frequency mappings and per-column medians for the five spending columns only from the current training fold. During `transform(X)`, it will use those mappings and medians; a group not observed in `X_train` receives `GroupSize = 1`. Spending values are filled from the learned medians before calculating `TotalSpend` and `HasSpending`, so missing values are never interpreted as zero. The transformer never receives or uses `y`.

Feature engineering runs before the preprocessing `ColumnTransformer`. Numerical and categorical imputers therefore learn their values only within the same training fold. The subsequent preprocessing will apply categorical imputation and one-hot encoding, plus numerical imputation and scaling/log transforms where appropriate.

## Error Handling and Validation

- Missing or malformed `Cabin` values produce missing parsed components, which are handled downstream by imputers.
- Missing spending values are filled only with medians learned from the current training fold before aggregated features are calculated; they are never silently converted to zero.
- Parsing is vectorized and preserves row order and index.
- Tests/checks must confirm that the row count is unchanged, all expected engineered columns exist, unseen validation groups map to size one, and no raw identifiers reach the estimator.

## Out of Scope

This step does not add a model, hyperparameter search, MLP, target encoding, or a Kaggle submission. Those belong to subsequent project stages.
