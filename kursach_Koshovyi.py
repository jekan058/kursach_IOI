"""
Особливості коду:
- Структурований підхід до EDA (попереднього аналізу даних).
- Використання пайплайнів (Pipeline) та ColumnTransformer для елегантної й гнучкої 
  попередньої обробки даних.
- Впровадження GridSearchCV для підбору гіперпараметрів і вибору найкращої моделі.
- Порівняння декількох моделей (LogisticRegression, RandomForest, GradientBoosting) з використанням 
  пайплайнів.
- Збереження результатів (метрик) у JSON файл.
- Використання логування для більш професійного підходу до розробки.
- Попередній контроль наявності датасету та опрацювання можливих проблем.

Після виконання скрипту:
- Будуть виведені результати EDA.
- Буде проведена автоматична підготовка даних.
- Виконано пошук найкращих гіперпараметрів для кожної моделі.
- Найкращі метрики та результати будуть виведені у консоль.
- Створено та збережено файл results.json з ключовими метриками найкращої моделі.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
import sys
import datetime

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RESULT_JSON = "results.json"


def load_data(path: str) -> pd.DataFrame:
    """Завантажити датасет та виконати базові перевірки."""
    try:
        data = pd.read_csv(path)
        logging.info("Датасет успішно завантажено.")
        return data
    except FileNotFoundError:
        logging.error(f"Файл {path} не знайдено. Завершення.")
        sys.exit(1)


def eda(data: pd.DataFrame) -> None:
    """Простий попередній аналіз даних (EDA)."""
    logging.info("Виконуємо EDA...")
    # Виведемо інформацію про датуфрейм
    print("Перші 5 рядків датасету:")
    print(data.head())
    print("\nІнформація про датафрейм:")
    data.info()
    print("\nСтатистичний опис числових ознак:")
    print(data.describe())
    print("\nРозподіл цільової змінної (Churn):")
    print(data['Churn'].value_counts())
    
    # Візуалізація цільової змінної
    plt.figure(figsize=(5,4))
    sns.countplot(x='Churn', data=data)
    plt.title("Розподіл відтоку клієнтів")
    plt.show()


def preprocess_data(data: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    """Попередня обробка даних: очистка, кодування, формування X та y."""
    
    # Перетворимо TotalCharges у числовий формат
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    # Заповнимо пропуски у TotalCharges середнім
    data['TotalCharges'] = SimpleImputer(strategy='mean').fit_transform(data[['TotalCharges']])

    # customerID - неінформативний для моделі
    if 'customerID' in data.columns:
        data.drop('customerID', axis=1, inplace=True)

    # Цільова змінна
    data['Churn'] = data['Churn'].map({'Yes':1, 'No':0})
    
    # Замінимо бінарні ознаки
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        data[col] = data[col].map({'Yes':1,'No':0})
    
    # Ознаки з "No internet service" -> "No"
    replace_cols = ['OnlineSecurity','OnlineBackup','DeviceProtection',
                    'TechSupport','StreamingTV','StreamingMovies']
    for col in replace_cols:
        data[col] = data[col].replace({'No internet service':'No'})
        
    X = data.drop('Churn', axis=1)
    y = data['Churn']

    return X, y


def build_preprocessing_pipeline(X: pd.DataFrame):
    """Повертає ColumnTransformer для попередньої обробки (імпутація, кодування, масштабування)."""
    
    # Розділимо ознаки на числові та категоріальні
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor


def build_models(preprocessor):
    """Створюємо словник моделей з пайплайнами та їх гіперпараметрами для пошуку."""
    # Логістична регресія
    pipe_lr = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    param_lr = {
        'clf__C': [0.1, 1, 10],
        'clf__penalty': ['l2']
    }

    # Random Forest
    pipe_rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(random_state=42))
    ])
    param_rf = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 5, 10],
        'clf__min_samples_split': [2, 5]
    }

    # Gradient Boosting
    pipe_gb = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', GradientBoostingClassifier(random_state=42))
    ])
    param_gb = {
        'clf__n_estimators': [100, 200],
        'clf__learning_rate': [0.05, 0.1],
        'clf__max_depth': [3, 5]
    }

    models_params = {
        'LogisticRegression': (pipe_lr, param_lr),
        'RandomForest': (pipe_rf, param_rf),
        'GradientBoosting': (pipe_gb, param_gb)
    }

    return models_params


def evaluate_model(model, X_test, y_test, model_name: str):
    """Обчислюємо метрики для моделі, виводимо та повертаємо метрики у вигляді словника."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, 'predict_proba') else None

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Побудова матриці
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix ({model_name})")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # ROC-AUC
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')
        plt.plot([0,1],[0,1],'r--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC-крива для {model_name}')
        plt.legend()
        plt.show()
    else:
        roc_auc = None

    metrics = {
        'model': model_name,
        'accuracy': report['accuracy'],
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1': report['1']['f1-score'],
        'roc_auc': roc_auc
    }

    return metrics


def main():
    data = load_data(DATA_PATH)
    eda(data)
    X, y = preprocess_data(data)

    # Розділяємо на тренувальну та тестову вибірки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessing_pipeline(X_train)
    models_params = build_models(preprocessor)

    # Налаштування перехресної валідації
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_metrics = None
    all_results = []

    # Пошук по сітці гіперпараметрів для кожної моделі
    for model_name, (pipe, params) in models_params.items():
        logging.info(f"Оптимізація гіперпараметрів для {model_name}...")
        grid_search = GridSearchCV(pipe, param_grid=params, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        logging.info(f"{model_name}: Найкращі гіперпараметри: {grid_search.best_params_}")
        best_model = grid_search.best_estimator_

        # Оцінка якості на тестовій вибірці
        metrics = evaluate_model(best_model, X_test, y_test, model_name)
        metrics['best_params'] = grid_search.best_params_
        all_results.append(metrics)

        # Зберігаємо найкращу модель за AUC
        if best_metrics is None or (metrics['roc_auc'] is not None and metrics['roc_auc'] > best_metrics['roc_auc']):
            best_metrics = metrics

    logging.info("Підсумкові результати:")
    for res in all_results:
        logging.info(f"{res['model']}: AUC={res['roc_auc']:.4f}, Acc={res['accuracy']:.4f}, F1={res['f1']:.4f}")

    # Збереження результатів у JSON
    best_metrics['timestamp'] = datetime.datetime.now().isoformat()
    with open(RESULT_JSON, 'w', encoding='utf-8') as f:
        json.dump(best_metrics, f, ensure_ascii=False, indent=4)

    logging.info(f"Найкраща модель: {best_metrics['model']}, результати збережено у {RESULT_JSON}.")


if __name__ == "__main__":
    main()
