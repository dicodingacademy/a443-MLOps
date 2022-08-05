"""
Author: Dicoding
Date: 5/8/2022
This is the eda_library.py module.
Usage:
- EDA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')

def import_data(pth):
    """Returns a pandas DataFrame for the CSV found at pth

    Args:
        pth (str): a path to the CSV file

    Returns:
        pandas DataFrame
    """
    return pd.read_csv(pth)


def perform_eda(dataframe):
    """Perform EDA on data_frame and save figures to images folder

    Args:
        dataframe (DataFrame): pandas DataFrame

    Returns:
        eda_df : pandas DataFrame
    """
    eda_df = dataframe.copy(deep=True)
    
    eda_df = eda_df[eda_df['TotalCharges'] != ' ']
    eda_df['TotalCharges'] = eda_df.TotalCharges.astype(float)
    
    churn = eda_df.Churn.value_counts()
    plt.figure(figsize=(7, 7))
    plt.pie(churn, labels=['No', 'Yes'], autopct='%1.1f%%')
    plt.title('Persentasi Churn Customer', loc='center', fontsize=18)
    plt.savefig('images/persentasi_churn.png')
    
    numerical_columns = ['MonthlyCharges', 'TotalCharges', 'tenure']
    fig, ax = plt.subplots(1, 3, figsize=(14, 7))
    eda_df[eda_df.Churn == 'No'][numerical_columns].hist(
        bins=20,
        color='blue',
        alpha=0.5,
        ax=ax
    )
    eda_df[eda_df.Churn == 'Yes'][numerical_columns].hist(
        bins=20,
        color='orange',
        alpha=0.5,
        ax=ax
    )
    plt.tight_layout()
    plt.savefig('images/distribusi_kolom_numerik.png')
    
    fig, ax = plt.subplots(3, 3, figsize=(14, 12))
    sns.countplot(data=eda_df, x='gender', hue='Churn', ax=ax[0][0])
    sns.countplot(data=eda_df, x='Partner', hue='Churn', ax=ax[0][1])
    sns.countplot(data=eda_df, x='SeniorCitizen', hue='Churn', ax=ax[0][2])
    sns.countplot(data=eda_df, x='PhoneService', hue='Churn', ax=ax[1][0])
    sns.countplot(data=eda_df, x='StreamingTV', hue='Churn', ax=ax[1][1])
    sns.countplot(data=eda_df, x='InternetService', hue='Churn', ax=ax[1][2])
    sns.countplot(data=eda_df, x='PaperlessBilling', hue='Churn', ax=ax[2][1])
    plt.tight_layout()
    plt.savefig('images/distribusi_kolom_kategorik.png')
    
    return eda_df


if __name__ == '__main__':
    customer_df = import_data(pth='./data/Telco-Customer-Churn.csv')
    
    eda_df = perform_eda(dataframe=customer_df)