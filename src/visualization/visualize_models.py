# -*- coding: utf-8 -*-

"""
Скрипт для визуализации результатов обученных моделей
"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.metrics import balanced_accuracy_score, precision_recall_curve
from sklearn.calibration import calibration_curve
import pickle
from tensorflow.keras.models import load_model
from visualization.model_visualizer import ModelVisualizer
import joblib
from src.data.data_loader import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns

def load_test_data():
    """Загрузка тестовых данных"""
    print("Загрузка данных...")
    
    # Загрузка сохраненных данных
    X_test = np.load('source/X_test.npy', allow_pickle=True)
    y_test = np.load('source/y_test.npy', allow_pickle=True)
    
    # Загрузка имен признаков
    with open('source/feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    print(f"\nРазмерность тестовых данных: {X_test.shape}")
    print(f"Количество признаков: {len(feature_names)}")
    print("\nПризнаки:")
    for i, name in enumerate(feature_names):
        print(f"{i+1}. {name}")
    
    return X_test, y_test, feature_names

def prepare_features(df):
    """Подготовка признаков в том же формате, что использовался при обучении."""
    # Базовые признаки
    features = pd.DataFrame()
    features['int_rate'] = df['int_rate']
    features['grade'] = pd.Categorical(df['grade']).codes
    features['sub_grade'] = pd.Categorical(df['sub_grade']).codes
    features['term'] = df['term'].apply(lambda x: int(x.split()[0]))
    
    # Расчет производных признаков с обработкой деления на ноль
    annual_inc_safe = df['annual_inc'].replace(0, np.nan)
    features['installment_to_income'] = (df['installment'] / annual_inc_safe).fillna(0)
    features['loan_to_income'] = (df['loan_amnt'] / annual_inc_safe).fillna(0)
    
    return features

def load_models():
    """Загрузка обученных моделей."""
    print("\nЗагрузка моделей...")
    models = {}
    
    # Загрузка Random Forest
    with open('models/random_forest_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    print(f"\nRandom Forest ожидает признаков: {len(rf_model.feature_names_in_)}")
    print("Имена признаков Random Forest:")
    for i, name in enumerate(rf_model.feature_names_in_, 1):
        print(f"{i}. {name}")
    models['Random Forest'] = rf_model
    
    # Загрузка XGBoost
    with open('models/xgboost_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    print(f"\nXGBoost ожидает признаков: {len(xgb_model.feature_names_in_)}")
    models['XGBoost'] = xgb_model
    
    # Загрузка нейронной сети
    nn_model = load_model('models/neural_network_model.h5')
    print(f"Нейронная сеть ожидает признаков: {nn_model.layers[0].input_shape[1]}")
    models['Neural Network'] = nn_model
    
    # Загрузка скейлера
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print(f"Скейлер ожидает признаков: {scaler.n_features_in_}")
    
    return models, scaler

def get_predictions(models, X_test, scaler):
    """Получение предсказаний от всех моделей."""
    X_scaled = scaler.transform(X_test)
    
    predictions = {}
    probabilities = {}
    
    for name, model in models.items():
        if name == 'Neural Network':
            probabilities[name] = model.predict(X_scaled)
            predictions[name] = (probabilities[name] > 0.5).astype(int)
        else:
            predictions[name] = model.predict(X_test)
            probabilities[name] = model.predict_proba(X_test)[:, 1]
    
    return predictions, probabilities

def calculate_metrics(predictions, probabilities, y_test):
    """Расчет метрик для всех моделей"""
    metrics = {}
    
    for name in predictions.keys():
        report = classification_report(y_test, predictions[name], output_dict=True)
        roc_auc = roc_auc_score(y_test, probabilities[name])
        
        metrics[name] = {
            'accuracy': report['accuracy'],
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1': report['weighted avg']['f1-score'],
            'roc_auc': roc_auc
        }
    
    return metrics

def plot_roc_curves(y_test, probabilities):
    """Построение ROC-кривых для всех моделей."""
    plt.figure(figsize=(10, 8))
    
    for name, probs in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривые для всех моделей')
    plt.legend(loc="lower right")
    
    # Сохранение графика
    plt.savefig('visualizations/roc_curves.png')
    plt.close()

def print_score(y_true, y_pred, train=True):
    """Вывод метрик качества модели."""
    dataset = "тренировочном" if train else "тестовом"
    print(f"\nМетрики на {dataset} наборе:")
    print("=" * 50)
    print(classification_report(y_true, y_pred))

def plot_confusion_matrices(models, X, y):
    """Построение матриц ошибок для всех моделей."""
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    for ax, (name, model) in zip(axes, models.items()):
        if name == 'Neural Network':
            y_pred = (model.predict(X) > 0.5).astype(int)
            disp = ConfusionMatrixDisplay.from_predictions(
                y, y_pred, ax=ax, cmap='Blues',
                values_format='d', display_labels=['Default', 'Fully-Paid']
            )
        else:
            disp = ConfusionMatrixDisplay.from_estimator(
                model, X, y, ax=ax, cmap='Blues',
                values_format='d', display_labels=['Default', 'Fully-Paid']
            )
        disp.ax_.set_title(f'Матрица ошибок - {name}')
    
    plt.tight_layout()
    plt.savefig('visualizations/confusion_matrices.png')
    plt.close()

def plot_roc_curves_sklearn(models, X, y):
    """Построение ROC-кривых с использованием sklearn."""
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        if name == 'Neural Network':
            y_pred_proba = model.predict(X)
            RocCurveDisplay.from_predictions(
                y, y_pred_proba,
                name=name
            )
        else:
            RocCurveDisplay.from_estimator(
                model, X, y,
                name=name
            )
    
    plt.title('ROC-кривые для всех моделей')
    plt.grid(True)
    plt.savefig('visualizations/roc_curves_sklearn.png')
    plt.close()

def plot_feature_importance(models, feature_names):
    """Визуализация важности признаков для Random Forest и XGBoost."""
    plt.figure(figsize=(12, 6))
    
    for name, model in models.items():
        if name in ['Random Forest', 'XGBoost']:
            # Получение важности признаков
            importance = model.feature_importances_
            # Создание DataFrame для сортировки
            feat_imp = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=True)
            
            # Построение графика
            plt.subplot(1, 2, 1 if name == 'Random Forest' else 2)
            plt.barh(range(len(feat_imp)), feat_imp['importance'])
            plt.yticks(range(len(feat_imp)), feat_imp['feature'])
            plt.xlabel('Важность')
            plt.title(f'Важность признаков - {name}')
    
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance.png', bbox_inches='tight')
    plt.close()

def plot_calibration_curves(models, X, y):
    """Построение калибровочных кривых для всех моделей."""
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        if name == 'Neural Network':
            prob_pos = model.predict(X)
        else:
            prob_pos = model.predict_proba(X)[:, 1]
        
        # Получение калибровочной кривой
        prob_true, prob_pred = calibration_curve(y, prob_pos, n_bins=10)
        
        # Построение графика
        plt.plot(prob_pred, prob_true, marker='o', label=name)
    
    # Добавление диагональной линии (идеальная калибровка)
    plt.plot([0, 1], [0, 1], 'k--', label='Идеальная калибровка')
    
    plt.xlabel('Средняя прогнозируемая вероятность')
    plt.ylabel('Доля положительных исходов')
    plt.title('Калибровочные кривые моделей')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('visualizations/calibration_curves.png')
    plt.close()

def calculate_additional_metrics(y_true, predictions, probabilities):
    """Расчет дополнительных метрик для несбалансированных классов."""
    metrics = {}
    
    for name in predictions.keys():
        # Сбалансированная точность
        balanced_acc = balanced_accuracy_score(y_true, predictions[name])
        
        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(y_true, probabilities[name])
        pr_auc = auc(recall, precision)
        
        metrics[name] = {
            'balanced_accuracy': balanced_acc,
            'pr_auc': pr_auc
        }
    
    return metrics

def main():
    """Основная функция"""
    print("Визуализация результатов моделей")
    print("=" * 50)
    
    # Загрузка данных
    print("Загрузка данных...")
    data_loader = DataLoader('source/lending_club_loan_two.csv')
    df = data_loader.load_data()
    
    # Проверка уникальных значений loan_status
    print("\nУникальные значения loan_status:")
    print(df['loan_status'].value_counts())
    
    # Подготовка признаков
    X = prepare_features(df)
    y = df['loan_status']  # Значения уже закодированы как 0 и 1
    
    print(f"\nРазмерность подготовленных данных: {X.shape}")
    print(f"Количество признаков: {X.shape[1]}")
    print("\nПризнаки:")
    for i, col in enumerate(X.columns, 1):
        print(f"{i}. {col}")
    
    # Загрузка моделей и получение предсказаний
    models, scaler = load_models()
    predictions, probabilities = get_predictions(models, X, scaler)
    
    # Вывод стандартных метрик для каждой модели
    print("\nРезультаты оценки моделей:")
    print("=" * 50)
    for name in models.keys():
        print(f"\nМодель: {name}")
        print_score(y, predictions[name], train=False)
    
    # Расчет дополнительных метрик
    additional_metrics = calculate_additional_metrics(y, predictions, probabilities)
    print("\nДополнительные метрики для несбалансированных классов:")
    print("=" * 50)
    for name, metrics in additional_metrics.items():
        print(f"\nМодель: {name}")
        print(f"Сбалансированная точность: {metrics['balanced_accuracy']:.4f}")
        print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    
    # Построение всех визуализаций
    plot_roc_curves(y, probabilities)
    print("\nГрафики ROC-кривых сохранены в visualizations/roc_curves.png")
    
    plot_roc_curves_sklearn(models, X, y)
    print("Графики ROC-кривых (sklearn) сохранены в visualizations/roc_curves_sklearn.png")
    
    plot_confusion_matrices(models, X, y)
    print("Матрицы ошибок сохранены в visualizations/confusion_matrices.png")
    
    plot_feature_importance(models, X.columns)
    print("График важности признаков сохранен в visualizations/feature_importance.png")
    
    plot_calibration_curves(models, X, y)
    print("Калибровочные кривые сохранены в visualizations/calibration_curves.png")

if __name__ == "__main__":
    main() 