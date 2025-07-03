# -*- coding: utf-8 -*-

"""
Модуль для загрузки и базовой обработки данных
"""

import pandas as pd
from typing import Tuple, Optional

class DataLoader:
    """Класс для загрузки и базовой обработки данных"""
    
    def __init__(self, data_path: str):
        """
        Инициализация загрузчика данных
        
        Args:
            data_path (str): Путь к файлу с данными
        """
        self.data_path = data_path
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Загрузка данных из CSV файла
        
        Returns:
            pd.DataFrame: Загруженный датафрейм
        """
        self.df = pd.read_csv(self.data_path)
        return self.df
    
    def get_feature_lists(self) -> Tuple[list, list]:
        """
        Получение списков числовых и категориальных признаков
        
        Returns:
            Tuple[list, list]: (числовые признаки, категориальные признаки)
        """
        numeric_features = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 
                          'dti', 'open_acc', 'pub_rec', 'revol_bal', 'total_acc']
        
        categorical_features = ['grade', 'home_ownership', 'verification_status', 
                              'purpose', 'term']
        
        return numeric_features, categorical_features
    
    def create_target(self) -> pd.Series:
        """
        Создание целевой переменной
        
        Returns:
            pd.Series: Целевая переменная (1 - "Fully Paid", 0 - "Charged Off")
        """
        if self.df is None:
            self.load_data()
        return (self.df['loan_status'] == "Fully Paid").astype(int)
    
    def get_features(self) -> pd.DataFrame:
        """
        Получение признаков для модели
        
        Returns:
            pd.DataFrame: Датафрейм с признаками
        """
        if self.df is None:
            self.load_data()
            
        numeric_features, categorical_features = self.get_feature_lists()
        features = numeric_features + categorical_features
        
        return self.df[features].copy() 