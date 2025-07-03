# -*- coding: utf-8 -*-

"""
Тесты для модуля предобработки данных
"""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing.preprocessor import DataPreprocessor

@pytest.fixture
def sample_data():
    """Создание тестовых данных"""
    data = {
        'numeric_1': [1.0, 2.0, 3.0, 4.0],
        'numeric_2': [10.0, 20.0, 30.0, 40.0],
        'category_1': ['A', 'B', 'A', 'C'],
        'category_2': ['X', 'Y', 'X', 'Z']
    }
    return pd.DataFrame(data)

@pytest.fixture
def target_data():
    """Создание тестовой целевой переменной"""
    return pd.Series([1, 0, 1, 0])

@pytest.fixture
def preprocessor(tmp_path):
    """Создание препроцессора с временной директорией для моделей"""
    return DataPreprocessor(models_dir=str(tmp_path))

def test_encode_categorical_features(preprocessor, sample_data):
    """Тест кодирования категориальных признаков"""
    categorical_features = ['category_1', 'category_2']
    encoded_data = preprocessor.encode_categorical_features(
        sample_data, categorical_features
    )
    
    # Проверяем, что данные закодированы
    assert encoded_data['category_1'].dtype == np.int64
    assert encoded_data['category_2'].dtype == np.int64
    
    # Проверяем, что энкодеры сохранены
    assert len(preprocessor.label_encoders) == len(categorical_features)
    assert all(col in preprocessor.label_encoders for col in categorical_features)

def test_scale_numeric_features(preprocessor, sample_data):
    """Тест масштабирования числовых признаков"""
    numeric_features = ['numeric_1', 'numeric_2']
    scaled_data = preprocessor.scale_numeric_features(
        sample_data, numeric_features
    )
    
    # Проверяем, что данные масштабированы
    for col in numeric_features:
        scaled_values = scaled_data[col]
        assert abs(scaled_values.mean()) < 1e-10  # Близко к 0
        assert abs(scaled_values.std() - 1.0) < 1e-10  # Близко к 1
    
    # Проверяем, что скейлер сохранен
    assert preprocessor.scaler is not None

def test_prepare_data(preprocessor, sample_data, target_data):
    """Тест полной подготовки данных"""
    numeric_features = ['numeric_1', 'numeric_2']
    categorical_features = ['category_1', 'category_2']
    
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(
        sample_data, target_data,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        test_size=0.5,
        random_state=42
    )
    
    # Проверяем размеры выборок
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) + len(X_test) == len(sample_data)
    
    # Проверяем, что все признаки обработаны
    assert all(col in X_train.columns for col in numeric_features + categorical_features)
    assert all(X_train[col].dtype == np.int64 for col in categorical_features)
    
    # Проверяем, что числовые признаки масштабированы
    for col in numeric_features:
        assert abs(X_train[col].mean()) < 1e-10
        assert abs(X_train[col].std() - 1.0) < 1e-10 