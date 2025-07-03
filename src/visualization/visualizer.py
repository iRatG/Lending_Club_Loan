# -*- coding: utf-8 -*-

"""
Модуль для создания визуализаций
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List

class LoanVisualizer:
    """Класс для создания визуализаций данных о кредитах"""
    
    def __init__(self, output_dir: str = 'visualizations'):
        """
        Инициализация визуализатора
        
        Args:
            output_dir (str): Директория для сохранения визуализаций
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Настройка стиля
        plt.style.use('seaborn')
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def plot_loan_status_distribution(self, df: pd.DataFrame) -> None:
        """
        Построение распределения статусов кредитов
        
        Args:
            df (pd.DataFrame): Датафрейм с данными
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='loan_status')
        plt.title('Распределение статусов кредитов')
        plt.savefig(self.output_dir / 'loan_status_distribution.png')
        plt.close()
    
    def plot_numeric_distributions(self, df: pd.DataFrame, 
                                 numeric_columns: List[str]) -> None:
        """
        Построение распределений числовых переменных
        
        Args:
            df (pd.DataFrame): Датафрейм с данными
            numeric_columns (List[str]): Список числовых колонок
        """
        n_cols = len(numeric_columns)
        n_rows = (n_cols - 1) // 3 + 1
        
        plt.figure(figsize=(15, 5 * n_rows))
        for i, column in enumerate(numeric_columns, 1):
            plt.subplot(n_rows, 3, i)
            sns.histplot(data=df, x=column, hue='loan_status', multiple="stack")
            plt.title(f'Распределение {column}')
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'numeric_distributions.png')
        plt.close()
    
    def plot_categorical_distributions(self, df: pd.DataFrame, 
                                    categorical_columns: List[str]) -> None:
        """
        Построение распределений категориальных переменных
        
        Args:
            df (pd.DataFrame): Датафрейм с данными
            categorical_columns (List[str]): Список категориальных колонок
        """
        plt.figure(figsize=(15, 10))
        for i, column in enumerate(categorical_columns, 1):
            plt.subplot(2, 3, i)
            df_temp = pd.crosstab(df[column], df['loan_status'], 
                                normalize='index') * 100
            df_temp.plot(kind='bar', stacked=True)
            plt.title(f'Распределение {column} по статусу кредита')
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'categorical_distributions.png')
        plt.close()
    
    def plot_correlation_matrix(self, df: pd.DataFrame) -> None:
        """
        Построение корреляционной матрицы
        
        Args:
            df (pd.DataFrame): Датафрейм с данными
        """
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        correlation_matrix = numeric_df.corr()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Корреляционная матрица числовых переменных')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_matrix.png')
        plt.close()
    
    def create_all_visualizations(self, df: pd.DataFrame, 
                                numeric_columns: List[str],
                                categorical_columns: List[str]) -> None:
        """
        Создание всех визуализаций
        
        Args:
            df (pd.DataFrame): Датафрейм с данными
            numeric_columns (List[str]): Список числовых колонок
            categorical_columns (List[str]): Список категориальных колонок
        """
        print("\n=== Создание визуализаций ===")
        
        print("\nСоздание графика распределения статусов кредитов...")
        self.plot_loan_status_distribution(df)
        
        print("Создание графиков распределения числовых переменных...")
        self.plot_numeric_distributions(df, numeric_columns)
        
        print("Создание графиков для категориальных переменных...")
        self.plot_categorical_distributions(df, categorical_columns)
        
        print("Создание корреляционной матрицы...")
        self.plot_correlation_matrix(df)
        
        print(f"\nВизуализации сохранены в директории '{self.output_dir}'") 