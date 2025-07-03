# -*- coding: utf-8 -*-

"""
Модуль для визуализации результатов работы моделей машинного обучения
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix
import pandas as pd
from typing import Dict, List, Tuple
import os

class ModelVisualizer:
    """Класс для создания визуализаций результатов работы моделей"""
    
    def __init__(self, output_dir: str = 'visualizations/model_analysis'):
        """
        Инициализация визуализатора
        
        Args:
            output_dir (str): Директория для сохранения графиков
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Настройка стиля графиков
        plt.style.use('seaborn')
        self.colors = ['#2ecc71', '#e74c3c', '#3498db']
    
    def plot_roc_curves(self, models_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
        """
        Построение ROC-кривых для всех моделей
        
        Args:
            models_data (Dict[str, Tuple]): Словарь с данными моделей {название: (y_true, y_pred_proba)}
        """
        plt.figure(figsize=(10, 8))
        
        for (name, (y_true, y_pred_proba)), color in zip(models_data.items(), self.colors):
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            plt.plot(fpr, tpr, label=name, color=color, linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-кривые для всех моделей')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.output_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrices(self, models_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
        """
        Построение матриц ошибок для каждой модели
        
        Args:
            models_data (Dict[str, Tuple]): Словарь с данными моделей {название: (y_true, y_pred)}
        """
        fig, axes = plt.subplots(1, len(models_data), figsize=(15, 5))
        
        for (name, (y_true, y_pred)), ax in zip(models_data.items(), axes):
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
            ax.set_title(f'Матрица ошибок\n{name}')
            ax.set_xlabel('Предсказанные значения')
            ax.set_ylabel('Истинные значения')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance(self, importance_data: Dict[str, pd.DataFrame]) -> None:
        """
        Построение графиков важности признаков
        
        Args:
            importance_data (Dict[str, pd.DataFrame]): Словарь с данными важности признаков
        """
        fig, axes = plt.subplots(1, len(importance_data), figsize=(15, 6))
        
        for (name, data), ax in zip(importance_data.items(), axes):
            sns.barplot(data=data, x='importance', y='feature', ax=ax)
            ax.set_title(f'Важность признаков\n{name}')
            ax.set_xlabel('Важность')
            ax.set_ylabel('Признак')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_nn_learning_curves(self, history: Dict[str, List[float]]) -> None:
        """
        Построение графиков обучения нейронной сети
        
        Args:
            history (Dict[str, List[float]]): История обучения модели
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # График точности
        ax1.plot(history['accuracy'], label='Обучающая выборка')
        ax1.plot(history['val_accuracy'], label='Проверочная выборка')
        ax1.set_title('График точности')
        ax1.set_xlabel('Эпоха')
        ax1.set_ylabel('Точность')
        ax1.legend()
        ax1.grid(True)
        
        # График функции потерь
        ax2.plot(history['loss'], label='Обучающая выборка')
        ax2.plot(history['val_loss'], label='Проверочная выборка')
        ax2.set_title('График функции потерь')
        ax2.set_xlabel('Эпоха')
        ax2.set_ylabel('Потери')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/nn_learning_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_model_comparison(self, metrics: Dict[str, Dict[str, float]]) -> None:
        """
        Построение сравнительной диаграммы метрик моделей
        
        Args:
            metrics (Dict[str, Dict[str, float]]): Словарь с метриками моделей
        """
        # Преобразование данных для построения
        df = pd.DataFrame(metrics).T
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(metrics))
        width = 0.15
        
        for i, metric in enumerate(df.columns):
            plt.bar(x + i * width, df[metric], width, label=metric)
        
        plt.xlabel('Модель')
        plt.ylabel('Значение')
        plt.title('Сравнение метрик моделей')
        plt.xticks(x + width * 2, df.index, rotation=45)
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close() 