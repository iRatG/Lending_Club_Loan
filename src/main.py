# -*- coding: utf-8 -*-

"""
Главный модуль для анализа кредитных данных
"""

import sys
from pathlib import Path
from src.data.data_loader import DataLoader
from src.preprocessing.preprocessor import DataPreprocessor
from src.models.model_trainer import ModelTrainer
from src.visualization.visualizer import LoanVisualizer
from src.analysis.feature_analyzer import FeatureAnalyzer

def run_data_loading(data_loader: DataLoader):
    """Загрузка данных"""
    print("\n=== Загрузка данных ===")
    df = data_loader.load_data()
    print(f"Загружено {len(df)} записей")
    return df

def run_visualization(df, visualizer: LoanVisualizer):
    """Создание базовых визуализаций"""
    print("\n=== Создание базовых визуализаций ===")
    numeric_features, categorical_features = data_loader.get_feature_lists()
    visualizer.create_all_visualizations(df, numeric_features, categorical_features)
    print("Визуализации сохранены в директории 'visualizations'")

def run_feature_analysis(df, analyzer: FeatureAnalyzer):
    """Углубленный анализ признаков"""
    print("\n=== Запуск углубленного анализа признаков ===")
    analyzer.run_full_analysis(df)
    print("\nАнализ завершен. Визуализации сохранены в директории 'visualizations/feature_analysis'")

def run_data_preparation(df, preprocessor: DataPreprocessor):
    """Подготовка данных"""
    print("\n=== Подготовка данных ===")
    
    print("Создание специфичных признаков для кредитов...")
    df = preprocessor.create_loan_features(df)
    
    print("Кодирование категориальных признаков...")
    df = preprocessor.encode_categorical_features(df)
    
    print("Отбор признаков...")
    X, y = preprocessor.select_features(df)
    
    print("Разделение данных на обучающую и тестовую выборки...")
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    print("Масштабирование числовых признаков...")
    X_train, X_test = preprocessor.scale_features(X_train, X_test)
    
    print("Подготовка данных завершена")
    return X_train, X_test, y_train, y_test, list(X.columns)

def run_model_training(X_train, X_test, y_train, y_test, feature_names, trainer: ModelTrainer):
    """Обучение и оценка моделей"""
    print("\n=== Обучение и оценка моделей ===")
    trainer.train_and_evaluate_all(X_train, X_test, y_train, y_test, feature_names)
    print("\nМодели сохранены в директории 'models'")

def run_model_visualization():
    """Визуализация результатов обученных моделей"""
    print("\n=== Визуализация результатов моделей ===")
    from src.visualization.visualize_models import main as visualize_models
    visualize_models()
    print("\nВизуализации моделей сохранены в директории 'visualizations'")

def run_analysis(steps):
    """Запуск выбранных этапов анализа"""
    data_loader = DataLoader('source/lending_club_loan_two.csv')
    visualizer = LoanVisualizer()
    analyzer = FeatureAnalyzer()
    preprocessor = DataPreprocessor()
    trainer = ModelTrainer()
    
    df = None
    if 1 in steps or any(step in steps for step in [2, 3, 4, 5]):
        df = run_data_loading(data_loader)
    
    if 2 in steps:
        run_visualization(df, visualizer)
    
    if 3 in steps:
        run_feature_analysis(df, analyzer)
    
    if 4 in steps or 5 in steps:
        X_train, X_test, y_train, y_test, feature_names = run_data_preparation(df, preprocessor)
    
    if 5 in steps:
        run_model_training(X_train, X_test, y_train, y_test, feature_names, trainer)
    
    if 6 in steps:
        run_model_visualization()

def main():
    """Основная функция"""
    print("Анализ кредитных данных")
    print("=" * 50 + "\n")
    
    print("Выберите этапы для выполнения (введите номера через пробел):")
    print("1. Загрузка данных")
    print("2. Создание базовых визуализаций")
    print("3. Углубленный анализ признаков")
    print("4. Подготовка данных")
    print("5. Обучение и оценка моделей")
    print("6. Визуализация результатов моделей")
    
    try:
        choice = input("\nВаш выбор: ")
        steps = [int(x) for x in choice.split()]
        run_analysis(steps)
    except ValueError:
        print("Ошибка: Введите корректные номера этапов")
        sys.exit(1)

if __name__ == "__main__":
    main() 