# -*- coding: utf-8 -*-

"""
Модуль для предобработки данных и инженерии признаков
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path
from typing import Dict, Tuple, List, Any

class DataPreprocessor:
    """Класс для предобработки данных и инженерии признаков"""
    
    def __init__(self, models_dir: str = 'models'):
        """
        Инициализация препроцессора
        
        Args:
            models_dir (str): Директория для сохранения моделей
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: StandardScaler = None
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                 categorical_features: List[str]) -> pd.DataFrame:
        """
        Кодирование категориальных признаков
        
        Args:
            df (pd.DataFrame): Датафрейм с данными
            categorical_features (List[str]): Список категориальных признаков
            
        Returns:
            pd.DataFrame: Датафрейм с закодированными признаками
        """
        df_encoded = df.copy()
        
        for feature in categorical_features:
            le = LabelEncoder()
            df_encoded[feature] = le.fit_transform(df_encoded[feature].astype(str))
            self.label_encoders[feature] = le
            
            # Сохранение энкодера
            with open(self.models_dir / f'{feature}_encoder.pkl', 'wb') as f:
                pickle.dump(le, f)
        
        return df_encoded
    
    def scale_numeric_features(self, df: pd.DataFrame, 
                             numeric_features: List[str]) -> pd.DataFrame:
        """
        Масштабирование числовых признаков с обработкой выбросов
        
        Args:
            df (pd.DataFrame): Датафрейм с данными
            numeric_features (List[str]): Список числовых признаков
            
        Returns:
            pd.DataFrame: Датафрейм с масштабированными признаками
        """
        df_scaled = df.copy()
        
        # Заменяем бесконечные значения на NaN
        df_scaled[numeric_features] = df_scaled[numeric_features].replace(
            [np.inf, -np.inf], np.nan
        )
        
        # Заполняем пропущенные значения медианой и обрабатываем выбросы
        for col in numeric_features:
            median_val = df_scaled[col].median()
            df_scaled[col] = df_scaled[col].fillna(median_val)
            
            # Ограничиваем значения 1 и 99 перцентилями
            p1 = df_scaled[col].quantile(0.01)
            p99 = df_scaled[col].quantile(0.99)
            df_scaled[col] = df_scaled[col].clip(p1, p99)
        
        # Стандартизация
        self.scaler = StandardScaler()
        df_scaled[numeric_features] = self.scaler.fit_transform(
            df_scaled[numeric_features]
        )
        
        # Сохранение скейлера
        with open(self.models_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        return df_scaled
    
    def create_date_features(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """
        Создание признаков на основе даты
        
        Args:
            df (pd.DataFrame): Датафрейм с данными
            date_col (str): Название колонки с датой
            
        Returns:
            pd.DataFrame: Датафрейм с новыми признаками
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        df[f'{date_col}_year'] = df[date_col].dt.year
        df[f'{date_col}_month'] = df[date_col].dt.month
        df[f'{date_col}_day'] = df[date_col].dt.day
        df[f'{date_col}_dayofweek'] = df[date_col].dt.dayofweek
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame, 
                                  feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Создание признаков взаимодействия
        
        Args:
            df (pd.DataFrame): Датафрейм с данными
            feature_pairs (List[Tuple[str, str]]): Список пар признаков
            
        Returns:
            pd.DataFrame: Датафрейм с новыми признаками
        """
        df = df.copy()
        
        for feat1, feat2 in feature_pairs:
            df[f'{feat1}_{feat2}_mult'] = df[feat1] * df[feat2]
            df[f'{feat1}_{feat2}_div'] = df[feat1] / (df[feat2] + 1e-6)
        
        return df
    
    def create_loan_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создание специфичных для кредитных данных признаков
        
        Args:
            df (pd.DataFrame): Датафрейм с данными
            
        Returns:
            pd.DataFrame: Датафрейм с новыми признаками
        """
        df = df.copy()
        
        # Отношение ежемесячного платежа к доходу
        df['installment_to_income'] = df['installment'] / (df['annual_inc'] / 12)
        
        # Отношение суммы кредита к доходу
        df['loan_to_income'] = df['loan_amnt'] / df['annual_inc']
        
        # Процент использования кредитной линии
        df['credit_utilization'] = df['revol_bal'] / (df['total_acc'] + 1)
        
        return df
    
    def select_features(self, df: pd.DataFrame, target_col: str,
                       threshold: float = 0.1) -> pd.DataFrame:
        """
        Отбор признаков на основе корреляции с целевой переменной
        
        Args:
            df (pd.DataFrame): Датафрейм с данными
            target_col (str): Название целевой переменной
            threshold (float): Пороговое значение корреляции
            
        Returns:
            pd.DataFrame: Датафрейм с отобранными признаками
        """
        df = df.copy()
        
        # Находим корреляции с целевой переменной
        correlations = df.corr()[target_col].abs()
        
        # Отбираем признаки с корреляцией выше порога
        selected_features = correlations[correlations > threshold].index.tolist()
        
        return df[selected_features]
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series,
                    numeric_features: List[str],
                    categorical_features: List[str],
                    date_features: List[str] = None,
                    interaction_pairs: List[Tuple[str, str]] = None,
                    create_loan_feats: bool = True,
                    feature_selection: bool = True,
                    test_size: float = 0.2,
                    random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                   pd.Series, pd.Series]:
        """
        Полная подготовка данных для обучения
        
        Args:
            X (pd.DataFrame): Признаки
            y (pd.Series): Целевая переменная
            numeric_features (List[str]): Список числовых признаков
            categorical_features (List[str]): Список категориальных признаков
            date_features (List[str]): Список признаков с датами
            interaction_pairs (List[Tuple[str, str]]): Пары признаков для взаимодействия
            create_loan_feats (bool): Создавать ли специфичные признаки для кредитов
            feature_selection (bool): Применять ли отбор признаков
            test_size (float): Размер тестовой выборки
            random_state (int): Случайное зерно
            
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        print("\n=== Подготовка данных ===")
        
        # Создание признаков на основе дат
        if date_features:
            print("\nСоздание признаков на основе дат...")
            for date_col in date_features:
                X = self.create_date_features(X, date_col)
        
        # Создание признаков взаимодействия
        if interaction_pairs:
            print("Создание признаков взаимодействия...")
            X = self.create_interaction_features(X, interaction_pairs)
        
        # Создание специфичных признаков для кредитов
        if create_loan_feats:
            print("Создание специфичных признаков для кредитов...")
            X = self.create_loan_features(X)
        
        # Кодирование категориальных признаков
        print("Кодирование категориальных признаков...")
        X = self.encode_categorical_features(X, categorical_features)
        
        # Отбор признаков
        if feature_selection:
            print("Отбор признаков...")
            X = pd.concat([X, y], axis=1)
            X = self.select_features(X, y.name, threshold=0.1)
            y = X[y.name]
            X = X.drop(columns=[y.name])
        
        # Разделение на обучающую и тестовую выборки
        print("Разделение данных на обучающую и тестовую выборки...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Масштабирование числовых признаков
        print("Масштабирование числовых признаков...")
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        X_train = self.scale_numeric_features(X_train, numeric_features)
        X_test[numeric_features] = self.scaler.transform(X_test[numeric_features])
        
        print("Подготовка данных завершена")
        
        return X_train, X_test, y_train, y_test 