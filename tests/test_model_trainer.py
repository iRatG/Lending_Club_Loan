# -*- coding: utf-8 -*-

"""
Тесты для модуля обучения модели
"""

import pytest
import pandas as pd
import numpy as np
from src.models.model_trainer import ModelTrainer

@pytest.fixture
def sample_data():
    """Создание тестовых данных"""
    np.random.seed(42)
    n_samples = 100
    
    X = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(0, 1, n_samples)
    })
    
    # Создаем синтетическую целевую переменную
    y = (X['feature_1'] + X['feature_2'] > 0).astype(int)
    
    # Разделяем на обучающую и тестовую выборки
    train_idx = np.random.choice([True, False], n_samples, p=[0.8, 0.2])
    
    return {
        'X_train': X[train_idx],
        'X_test': X[~train_idx],
        'y_train': y[train_idx],
        'y_test': y[~train_idx]
    }

@pytest.fixture
def model_trainer(tmp_path):
    """Создание тренера моделей с временной директорией"""
    return ModelTrainer(models_dir=str(tmp_path))

def test_train_model(model_trainer, sample_data):
    """Тест обучения модели"""
    model = model_trainer.train_model(
        sample_data['X_train'],
        sample_data['y_train']
    )
    
    # Проверяем, что модель обучена
    assert model is not None
    assert model_trainer.model is not None
    
    # Проверяем, что модель может делать предсказания
    predictions = model.predict(sample_data['X_test'])
    assert len(predictions) == len(sample_data['X_test'])
    assert set(predictions).issubset({0, 1})

def test_evaluate_model(model_trainer, sample_data):
    """Тест оценки модели"""
    # Сначала обучаем модель
    model_trainer.train_model(
        sample_data['X_train'],
        sample_data['y_train']
    )
    
    # Оцениваем модель
    report, conf_matrix = model_trainer.evaluate_model(
        sample_data['X_test'],
        sample_data['y_test']
    )
    
    # Проверяем формат отчета
    assert isinstance(report, str)
    assert 'precision' in report.lower()
    assert 'recall' in report.lower()
    
    # Проверяем матрицу ошибок
    assert isinstance(conf_matrix, np.ndarray)
    assert conf_matrix.shape == (2, 2)  # Бинарная классификация

def test_get_feature_importance(model_trainer, sample_data):
    """Тест получения важности признаков"""
    # Обучаем модель
    model_trainer.train_model(
        sample_data['X_train'],
        sample_data['y_train']
    )
    
    # Получаем важность признаков
    importance = model_trainer.get_feature_importance(
        sample_data['X_train'].columns
    )
    
    # Проверяем формат результата
    assert isinstance(importance, pd.DataFrame)
    assert 'feature' in importance.columns
    assert 'importance' in importance.columns
    assert len(importance) == len(sample_data['X_train'].columns)
    
    # Проверяем, что важности неотрицательные и сумма близка к 1
    assert all(importance['importance'] >= 0)
    assert abs(importance['importance'].sum() - 1.0) < 1e-10

def test_train_and_evaluate(model_trainer, sample_data):
    """Тест полного процесса обучения и оценки"""
    model_trainer.train_and_evaluate(
        sample_data['X_train'],
        sample_data['X_test'],
        sample_data['y_train'],
        sample_data['y_test']
    )
    
    # Проверяем, что модель обучена
    assert model_trainer.model is not None
    
    # Проверяем, что модель сохранена
    assert (model_trainer.models_dir / 'random_forest_model.pkl').exists()

def test_model_error_handling(model_trainer, sample_data):
    """Тест обработки ошибок"""
    # Проверяем, что вызов evaluate_model без обученной модели вызывает ошибку
    with pytest.raises(ValueError):
        model_trainer.evaluate_model(
            sample_data['X_test'],
            sample_data['y_test']
        )
    
    # Проверяем, что вызов get_feature_importance без обученной модели вызывает ошибку
    with pytest.raises(ValueError):
        model_trainer.get_feature_importance(
            sample_data['X_train'].columns
        ) 