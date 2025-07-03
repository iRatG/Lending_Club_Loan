# -*- coding: utf-8 -*-

"""
Основной модуль для запуска анализа данных
"""

from data.data_loader import DataLoader
from visualization.visualizer import LoanVisualizer
from preprocessing.preprocessor import DataPreprocessor
from models.model_trainer import ModelTrainer

def run_analysis(steps: str) -> None:
    """
    Запуск анализа данных
    
    Args:
        steps (str): Строка с номерами этапов для выполнения
    """
    # Инициализация компонентов
    data_loader = DataLoader("source/lending_club_loan_two.csv")
    visualizer = LoanVisualizer()
    preprocessor = DataPreprocessor()
    model_trainer = ModelTrainer()
    
    # Выполнение выбранных этапов
    if '1' in steps:
        print("\n=== Этап 1: Загрузка данных ===")
        df = data_loader.load_data()
        print(f"Размер датасета: {df.shape}")
    
    if '2' in steps:
        if not '1' in steps:
            df = data_loader.load_data()
        
        numeric_features, categorical_features = data_loader.get_feature_lists()
        visualizer.create_all_visualizations(df, numeric_features, categorical_features)
    
    if '3' in steps or '4' in steps:
        if not '1' in steps:
            df = data_loader.load_data()
        
        # Подготовка данных
        X = data_loader.get_features()
        y = data_loader.create_target()
        
        numeric_features, categorical_features = data_loader.get_feature_lists()
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(
            X, y, numeric_features, categorical_features
        )
        
        if '4' in steps:
            # Обучение и оценка модели
            model_trainer.train_and_evaluate(X_train, X_test, y_train, y_test)

def main():
    """Основная функция"""
    print("Анализ кредитных данных")
    print("=" * 50)
    
    steps = input("""
Выберите этапы для выполнения (введите номера через пробел):
1. Загрузка данных
2. Создание визуализаций
3. Подготовка данных
4. Обучение и оценка модели
5. Выполнить все этапы

Ваш выбор: """)
    
    if '5' in steps or 'all' in steps.lower():
        steps = '1 2 3 4'
    
    run_analysis(steps)

if __name__ == "__main__":
    main() 