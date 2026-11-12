# Порівняння класичних моделей класифікації

## Постановка задачі
За клінічними показниками пацієнта визначити, чи є ознаки серцевого захворювання. Це бінарна класифікація, позитивний клас 1 (ознаки є). Основна метрика F1 для класу 1, додатково Recall і ROC AUC. Проєкт навчальний і не є медичним діагнозом.

## Набір даних
Heart Disease (UCI), файл `data/heart_disease.csv`: 303 об'єкти, 13 ознак (5 числових і 8 категоріальних), цільова змінна `target` (0 або 1), 6 пропусків (ca і thal).

## Методика експерименту
- фіксований train/test split 80/20 (`stratify=y`, `random_state=42`);
- підбір моделі та параметрів лише на train;
- єдина 5-кратна StratifiedKFold для всіх моделей;
- одноразове фінальне оцінювання на test.

## Підготовка даних
- SimpleImputer (медіана для числових, найчастіше значення для категоріальних);
- StandardScaler або MinMaxScaler для числових ознак;
- OneHotEncoder для категоріальних;
- ColumnTransformer і Pipeline (усе навчається лише на train-фолдах).

## Досліджені моделі
- Logistic Regression (L1 і L2, C від 0,01 до 100);
- k-NN (k, StandardScaler/MinMaxScaler, відстань Мінковського p від 1 до 4);
- Gaussian Naive Bayes;
- LDA;
- QDA (з reg_param);
- SVM (лінійна та RBF, GridSearchCV).

## Порівняння за крос-валідацією
| Модель | Параметри | CV F1 | CV F1 std | CV Recall | CV ROC AUC |
|---|---|---:|---:|---:|---:|
| Logistic Regression | L2, C=10 | 0,8363 | 0,0127 | 0,8016 | 0,9002 |
| k-NN | MinMaxScaler, p=1, k=11 | 0,8291 | 0,0375 | 0,8190 | 0,8971 |
| SVM | RBF, C=10, gamma=0,01 | 0,8270 | 0,0330 | 0,7830 | 0,8968 |
| LDA | - | 0,8128 | 0,0257 | 0,7656 | 0,8993 |
| QDA | reg_param=0,7 | 0,8022 | 0,0451 | 0,7561 | 0,8848 |
| GaussianNB | - | 0,7495 | 0,0215 | 0,8458 | 0,8773 |

## Фінальна модель і test-результат
Фінальна модель k-NN (MinMaxScaler, p=1, k=11). Правило, задане до відкриття test: серед моделей, чий CV F1 не нижчий за (найкращий F1 мінус його std), обирається та, що має найбільший Recall, бо пропустити хворого гірше.

Test (61 об'єкт): Accuracy 0,8689, Precision 0,8333, Recall 0,8929, F1 0,8621, ROC AUC 0,9524. Test не гірший за CV, але вибірка мала, тому оцінка груба.

## Структура репозиторію
```
classification-mini-project/
  data/heart_disease.csv
  notebooks/PR04_Chirva_Daniil.ipynb
  models/best_classifier.joblib
  reports/cv_model_comparison.csv
  reports/final_test_metrics.csv
  README.md
```

## Запуск
Відкрити Notebook у Google Colab, покласти `heart_disease.csv` у папку `data` і послідовно виконати всі комірки.
