# -*- coding: utf-8 -*-

"""
Тесты для модуля визуализации
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.visualization.visualizer import LoanVisualizer

@pytest.fixture
def sample_data():
    """Создание тестовых данных"""
    np.random.seed(42)
    n_samples = 100
    
    return pd.DataFrame({
        'loan_status': np.random.choice(['Fully Paid', 'Charged Off'], n_samples),
        'loan_amnt': np.random.normal(10000, 5000, n_samples),
        'int_rate': np.random.uniform(5, 25, n_samples),
        'grade': np.random.choice(['A', 'B', 'C'], n_samples),
        'home_ownership': np.random.choice(['RENT', 'OWN', 'MORTGAGE'], n_samples)
    })

@pytest.fixture
def visualizer(tmp_path):
    """Создание визуализатора с временной директорией"""
    return LoanVisualizer(output_dir=str(tmp_path))

def test_plot_loan_status_distribution(visualizer, sample_data):
    """Тест построения распределения статусов кредитов"""
    visualizer.plot_loan_status_distribution(sample_data)
    
    # Проверяем, что файл создан
    assert (visualizer.output_dir / 'loan_status_distribution.png').exists()

def test_plot_numeric_distributions(visualizer, sample_data):
    """Тест построения распределений числовых переменных"""
    numeric_columns = ['loan_amnt', 'int_rate']
    visualizer.plot_numeric_distributions(sample_data, numeric_columns)
    
    # Проверяем, что файл создан
    assert (visualizer.output_dir / 'numeric_distributions.png').exists()

def test_plot_categorical_distributions(visualizer, sample_data):
    """Тест построения распределений категориальных переменных"""
    categorical_columns = ['grade', 'home_ownership']
    visualizer.plot_categorical_distributions(sample_data, categorical_columns)
    
    # Проверяем, что файл создан
    assert (visualizer.output_dir / 'categorical_distributions.png').exists()

def test_plot_correlation_matrix(visualizer, sample_data):
    """Тест построения корреляционной матрицы"""
    visualizer.plot_correlation_matrix(sample_data)
    
    # Проверяем, что файл создан
    assert (visualizer.output_dir / 'correlation_matrix.png').exists()

def test_create_all_visualizations(visualizer, sample_data):
    """Тест создания всех визуализаций"""
    numeric_columns = ['loan_amnt', 'int_rate']
    categorical_columns = ['grade', 'home_ownership']
    
    visualizer.create_all_visualizations(
        sample_data,
        numeric_columns,
        categorical_columns
    )
    
    # Проверяем, что все файлы созданы
    expected_files = [
        'loan_status_distribution.png',
        'numeric_distributions.png',
        'categorical_distributions.png',
        'correlation_matrix.png'
    ]
    
    for file_name in expected_files:
        assert (visualizer.output_dir / file_name).exists()

def test_output_directory_creation(tmp_path):
    """Тест создания директории для вывода"""
    output_dir = tmp_path / "new_dir"
    visualizer = LoanVisualizer(output_dir=str(output_dir))
    
    assert output_dir.exists()
    assert output_dir.is_dir() 