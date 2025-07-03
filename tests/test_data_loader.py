# -*- coding: utf-8 -*-

"""
Тесты для модуля загрузки данных
"""

import pytest
import pandas as pd
import numpy as np
from src.data.data_loader import DataLoader

@pytest.fixture
def sample_data():
    """Создание тестовых данных"""
    data = {
        'loan_status': ['Fully Paid', 'Charged Off', 'Fully Paid'],
        'loan_amnt': [1000, 2000, 3000],
        'int_rate': [5.0, 7.0, 6.0],
        'grade': ['A', 'B', 'A'],
        'home_ownership': ['RENT', 'OWN', 'MORTGAGE']
    }
    return pd.DataFrame(data)

@pytest.fixture
def data_loader(tmp_path):
    """Создание тестового файла и инициализация DataLoader"""
    # Создаем тестовый CSV файл
    data = sample_data()
    csv_path = tmp_path / "test_loan_data.csv"
    data.to_csv(csv_path, index=False)
    
    return DataLoader(str(csv_path))

def test_load_data(data_loader, sample_data):
    """Тест загрузки данных"""
    df = data_loader.load_data()
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(sample_data)
    assert all(col in df.columns for col in sample_data.columns)

def test_get_feature_lists(data_loader):
    """Тест получения списков признаков"""
    numeric_features, categorical_features = data_loader.get_feature_lists()
    
    assert isinstance(numeric_features, list)
    assert isinstance(categorical_features, list)
    assert 'loan_amnt' in numeric_features
    assert 'grade' in categorical_features

def test_create_target(data_loader):
    """Тест создания целевой переменной"""
    data_loader.load_data()
    target = data_loader.create_target()
    
    assert isinstance(target, pd.Series)
    assert target.dtype == np.int64
    assert set(target.unique()) == {0, 1}

def test_get_features(data_loader):
    """Тест получения признаков"""
    data_loader.load_data()
    features = data_loader.get_features()
    
    assert isinstance(features, pd.DataFrame)
    numeric_features, categorical_features = data_loader.get_feature_lists()
    expected_columns = numeric_features + categorical_features
    assert all(col in features.columns for col in expected_columns) 