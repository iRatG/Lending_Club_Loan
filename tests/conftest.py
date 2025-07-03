# -*- coding: utf-8 -*-

"""
Конфигурация для тестов pytest
"""

import pytest
import os
import shutil

@pytest.fixture(autouse=True)
def run_around_tests(tmp_path):
    """
    Фикстура для подготовки и очистки временных файлов
    
    Args:
        tmp_path: Временная директория (предоставляется pytest)
    """
    # Подготовка перед каждым тестом
    os.makedirs(tmp_path / "models", exist_ok=True)
    os.makedirs(tmp_path / "visualizations", exist_ok=True)
    
    yield
    
    # Очистка после каждого теста
    if (tmp_path / "models").exists():
        shutil.rmtree(tmp_path / "models")
    if (tmp_path / "visualizations").exists():
        shutil.rmtree(tmp_path / "visualizations") 