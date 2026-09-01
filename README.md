# Spaceship Titanic

Проект по исследованию датасета соревнования **Spaceship Titanic**: EDA,
подготовка данных, создание признаков, сравнение моделей, подбор
гиперпараметров LightGBM, stacking и создание submission для Kaggle.

Основной файл проекта — [Research.ipynb](Research.ipynb).

## Датасет

Используется датасет соревнования
[Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic).
Локальные копии файлов находятся в каталоге `data/`.

| Выборка | Размер | Описание |
|---|---:|---|
| `train.csv` | 8 693 × 14 | Обучающая выборка с целевой переменной `Transported` |
| `test.csv` | 4 277 × 13 | Тестовая выборка Kaggle без целевой переменной |

Задача — бинарная классификация: предсказать, был ли пассажир перенесён в
другое измерение. Основная метрика соревнования — Accuracy.

Классы в train практически сбалансированы: `True` составляет 50.36%, а
`False` — 49.64%. Поэтому специальные методы балансировки классов не
используются.

## Выполненные этапы

1. Исследовательский анализ данных: распределение target, пропуски,
   дубликаты, категориальные и числовые признаки.
2. Feature engineering на основе идентификатора пассажира, каюты, расходов,
   возраста и количества пропусков.
3. Сравнение baseline и моделей Random Forest, Logistic Regression,
   Gradient Boosting, XGBoost и LightGBM.
4. Подбор гиперпараметров LightGBM через `RandomizedSearchCV`.
5. Построение stacking-ансамбля с Logistic Regression как мета-моделью.
6. Создание финального submission для Kaggle.

## Созданные признаки

Из `PassengerId`, `Cabin`, возраста и расходов получены следующие признаки:

- `GroupMemberNumber`, `GroupSize`, `IsAlone`;
- `CabinDeck`, `CabinNumber`, `CabinSide`;
- `TotalSpend`, `HasSpending`;
- `IsChild`;
- `MissingCount`.

Числовые пропуски заполняются медианой, категориальные — наиболее частым
значением. Эти операции, а также One-Hot Encoding, выполняются внутри
`Pipeline`, поэтому статистики вычисляются только на обучающей части каждого
фолда и не возникает утечки данных.

## Результаты одиночных моделей

Все метрики получены на обучающей выборке с помощью стратифицированной
пятикратной кросс-валидации и метрики Accuracy.

| Модель | Средняя Accuracy | Стандартное отклонение |
|---|---:|---:|
| Baseline Random Forest | 0.7849 | 0.0049 |
| Random Forest + Feature Engineering | 0.8003 | 0.0095 |
| Logistic Regression + Feature Engineering | 0.7914 | 0.0055 |
| Gradient Boosting | 0.8070 | 0.0059 |
| XGBoost | 0.8072 | 0.0063 |
| LightGBM | 0.8070 | 0.0087 |
| Tuned LightGBM | 0.8115 | 0.0074 |

Feature engineering улучшил Random Forest на 0.0154 Accuracy по сравнению с
baseline. Среди одиночных моделей лучший результат показал настроенный
LightGBM.

## Подбор гиперпараметров LightGBM

Для LightGBM использован `RandomizedSearchCV` с 30 случайными комбинациями
параметров, пятикратной стратифицированной кросс-валидацией и метрикой
Accuracy. Лучшая конфигурация показала среднюю Accuracy `0.8115`.

Ключевые параметры лучшей модели:

- `n_estimators = 700`;
- `learning_rate = 0.1`;
- `num_leaves = 7`;
- `min_child_samples = 20`;
- `colsample_bytree = 0.7`;
- `reg_alpha = 0.1` и `reg_lambda = 0.01`.

## Stacking

В ансамбль включены четыре базовые модели:

- Gradient Boosting;
- XGBoost;
- настроенный LightGBM;
- Random Forest после feature engineering.

Внутри `StackingClassifier` базовые модели создают out-of-fold вероятности
класса `Transported=True`. Для каждого объекта формируются четыре
мета-признака — по одному от каждой базовой модели. На этих признаках
обучается Logistic Regression.

На общей holdout-выборке stacking превзошёл tuned LightGBM:

| Модель | Accuracy |
|---|---:|
| Tuned LightGBM | 0.8074 |
| Stacking | 0.8183 |

Поэтому stacking выбран финальным решением.

## Kaggle submission

Финальный ансамбль переобучается на всей обучающей выборке. Предсказания для
test сохраняются в [submission_stacking.csv](submission_stacking.csv) с двумя
колонками: `PassengerId` и `Transported`.

На Kaggle submission занял **399 место из 1 633**, что соответствует
**топ-25%** участников.

## Структура проекта

```text
.
├── README.md
├── Research.ipynb
├── submission_stacking.csv
├── Leaderboard.png
└── data/
    ├── train.csv
    ├── test.csv
    └── sample_submission.csv
```

## Запуск проекта

Установите необходимые библиотеки:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm jupyter
```

Затем откройте ноутбук:

```bash
jupyter notebook Research.ipynb
```

Ячейки следует выполнять сверху вниз. Для воспроизведения финального файла
submission выполните последний раздел ноутбука.
