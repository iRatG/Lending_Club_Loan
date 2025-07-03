# -*- coding: utf-8 -*-

"""
Модуль для углубленного анализа признаков кредитных данных
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

class FeatureAnalyzer:
    """Класс для углубленного анализа признаков"""
    
    def __init__(self, output_dir: str = 'visualizations/feature_analysis'):
        """
        Инициализация анализатора признаков
        
        Args:
            output_dir (str): Директория для сохранения визуализаций
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Настройка стиля
        plt.style.use('seaborn')
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def analyze_credit_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Анализ метрик кредитоспособности (dti, open_acc, revol_bal, revol_util, total_acc)
        
        Args:
            df (pd.DataFrame): Датафрейм с данными
            
        Returns:
            Dict[str, float]: Статистики по метрикам
        """
        metrics = ['dti', 'open_acc', 'revol_bal', 'revol_util', 'total_acc']
        
        # Статистический анализ
        stats = df[metrics].describe()
        
        # Корреляция с целевой переменной
        correlations = df[metrics + ['loan_status']].corr()['loan_status']
        
        # Визуализация распределений
        plt.figure(figsize=(15, 10))
        for i, metric in enumerate(metrics, 1):
            plt.subplot(2, 3, i)
            sns.boxplot(data=df, x='loan_status', y=metric)
            plt.title(f'Распределение {metric} по статусу кредита')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'credit_metrics_distribution.png')
        plt.close()
        
        return {
            'statistics': stats,
            'correlations': correlations
        }
    
    def analyze_dates(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Анализ дат (issue_d, earliest_cr_line)
        
        Args:
            df (pd.DataFrame): Датафрейм с данными
            
        Returns:
            Dict[str, pd.Series]: Статистики по датам
        """
        # Преобразование дат
        df['issue_d'] = pd.to_datetime(df['issue_d'])
        df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'])
        
        # Расчет кредитной истории в годах
        df['credit_history_years'] = (df['issue_d'] - df['earliest_cr_line']).dt.days / 365
        
        # Анализ по годам
        yearly_stats = df.groupby(df['issue_d'].dt.year)['loan_status'].value_counts(normalize=True)
        
        # Визуализация
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        sns.histplot(data=df, x='credit_history_years', hue='loan_status')
        plt.title('Распределение длительности кредитной истории')
        
        plt.subplot(1, 2, 2)
        sns.lineplot(data=df, x=df['issue_d'].dt.year, y='loan_status')
        plt.title('Динамика статусов кредитов по годам')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dates_analysis.png')
        plt.close()
        
        return {
            'yearly_stats': yearly_stats,
            'credit_history_stats': df['credit_history_years'].describe()
        }
    
    def analyze_employment(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Анализ занятости (emp_title, emp_length)
        
        Args:
            df (pd.DataFrame): Датафрейм с данными
            
        Returns:
            Dict[str, pd.Series]: Статистики по занятости
        """
        # Топ профессий
        top_titles = df['emp_title'].value_counts().head(10)
        
        # Статистика по стажу
        length_stats = df.groupby('emp_length')['loan_status'].value_counts(normalize=True)
        
        # Визуализация
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        top_titles.plot(kind='bar')
        plt.title('Топ-10 профессий')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        sns.barplot(data=df, x='emp_length', y='loan_status')
        plt.title('Статус кредита по стажу работы')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'employment_analysis.png')
        plt.close()
        
        return {
            'top_titles': top_titles,
            'length_stats': length_stats
        }
    
    def analyze_financial_metrics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Анализ финансовых показателей (int_rate, annual_inc)
        
        Args:
            df (pd.DataFrame): Датафрейм с данными
            
        Returns:
            Dict[str, pd.Series]: Статистики по финансовым показателям
        """
        # Статистики
        int_rate_stats = df.groupby('loan_status')['int_rate'].describe()
        income_stats = df.groupby('loan_status')['annual_inc'].describe()
        
        # Визуализация
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        sns.boxplot(data=df, x='loan_status', y='int_rate')
        plt.title('Процентная ставка по статусу кредита')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(data=df, x='loan_status', y='annual_inc')
        plt.title('Годовой доход по статусу кредита')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'financial_metrics_analysis.png')
        plt.close()
        
        return {
            'int_rate_stats': int_rate_stats,
            'income_stats': income_stats
        }
    
    def analyze_loan_characteristics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Анализ характеристик кредита (term, home_ownership, verification_status, purpose)
        
        Args:
            df (pd.DataFrame): Датафрейм с данными
            
        Returns:
            Dict[str, pd.Series]: Статистики по характеристикам кредита
        """
        # Статистики
        term_stats = df.groupby('term')['loan_status'].value_counts(normalize=True)
        home_stats = df.groupby('home_ownership')['loan_status'].value_counts(normalize=True)
        verif_stats = df.groupby('verification_status')['loan_status'].value_counts(normalize=True)
        purpose_stats = df.groupby('purpose')['loan_status'].value_counts(normalize=True)
        
        # Визуализация
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        sns.barplot(data=df, x='term', y='loan_status')
        plt.title('Статус кредита по сроку')
        
        plt.subplot(2, 2, 2)
        sns.barplot(data=df, x='home_ownership', y='loan_status')
        plt.title('Статус кредита по типу владения жильем')
        
        plt.subplot(2, 2, 3)
        sns.barplot(data=df, x='verification_status', y='loan_status')
        plt.title('Статус кредита по верификации')
        
        plt.subplot(2, 2, 4)
        purpose_default = df.groupby('purpose')['loan_status'].mean()
        purpose_default.sort_values().plot(kind='bar')
        plt.title('Процент дефолтов по цели кредита')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'loan_characteristics_analysis.png')
        plt.close()
        
        return {
            'term_stats': term_stats,
            'home_stats': home_stats,
            'verif_stats': verif_stats,
            'purpose_stats': purpose_stats
        }
    
    def analyze_grade(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Анализ грейдов (grade, sub_grade)
        
        Args:
            df (pd.DataFrame): Датафрейм с данными
            
        Returns:
            Dict[str, pd.Series]: Статистики по грейдам
        """
        # Статистики
        grade_stats = df.groupby('grade')['loan_status'].value_counts(normalize=True)
        subgrade_stats = df.groupby('sub_grade')['loan_status'].value_counts(normalize=True)
        
        # Средние процентные ставки
        grade_rates = df.groupby('grade')['int_rate'].mean()
        subgrade_rates = df.groupby('sub_grade')['int_rate'].mean()
        
        # Визуализация
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        sns.barplot(data=df, x='grade', y='loan_status')
        plt.title('Статус кредита по грейду')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(data=df, x='grade', y='int_rate')
        plt.title('Процентная ставка по грейду')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'grade_analysis.png')
        plt.close()
        
        return {
            'grade_stats': grade_stats,
            'subgrade_stats': subgrade_stats,
            'grade_rates': grade_rates,
            'subgrade_rates': subgrade_rates
        }
    
    def analyze_loan_amounts(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Анализ сумм кредита (loan_amnt, installment)
        
        Args:
            df (pd.DataFrame): Датафрейм с данными
            
        Returns:
            Dict[str, pd.Series]: Статистики по суммам
        """
        # Статистики
        amount_stats = df.groupby('loan_status')['loan_amnt'].describe()
        installment_stats = df.groupby('loan_status')['installment'].describe()
        
        # Расчет отношения платежа к сумме кредита
        df['payment_ratio'] = df['installment'] / df['loan_amnt']
        
        # Визуализация
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        sns.boxplot(data=df, x='loan_status', y='loan_amnt')
        plt.title('Сумма кредита по статусу')
        
        plt.subplot(1, 3, 2)
        sns.boxplot(data=df, x='loan_status', y='installment')
        plt.title('Ежемесячный платеж по статусу')
        
        plt.subplot(1, 3, 3)
        sns.boxplot(data=df, x='loan_status', y='payment_ratio')
        plt.title('Отношение платежа к сумме кредита')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'loan_amounts_analysis.png')
        plt.close()
        
        return {
            'amount_stats': amount_stats,
            'installment_stats': installment_stats,
            'payment_ratio_stats': df['payment_ratio'].describe()
        }
    
    def run_full_analysis(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Запуск полного анализа всех групп признаков
        
        Args:
            df (pd.DataFrame): Датафрейм с данными
            
        Returns:
            Dict[str, Dict]: Результаты анализа по всем группам
        """
        print("\n=== Запуск углубленного анализа признаков ===")
        
        results = {}
        
        print("\nАнализ метрик кредитоспособности...")
        results['credit_metrics'] = self.analyze_credit_metrics(df)
        
        print("Анализ дат...")
        results['dates'] = self.analyze_dates(df)
        
        print("Анализ занятости...")
        results['employment'] = self.analyze_employment(df)
        
        print("Анализ финансовых показателей...")
        results['financial'] = self.analyze_financial_metrics(df)
        
        print("Анализ характеристик кредита...")
        results['loan_characteristics'] = self.analyze_loan_characteristics(df)
        
        print("Анализ грейдов...")
        results['grades'] = self.analyze_grade(df)
        
        print("Анализ сумм кредита...")
        results['amounts'] = self.analyze_loan_amounts(df)
        
        print(f"\nАнализ завершен. Визуализации сохранены в директории '{self.output_dir}'")
        
        return results 