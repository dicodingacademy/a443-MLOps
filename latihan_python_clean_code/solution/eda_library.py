"""
Author: Dicoding
Date: 5/8/2022
This is the eda_library.py module.
Usage:
- EDA
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')


def load_data(path):
    """Returns a pandas DataFrame for the CSV found at path

    Args:
        path (str): a path to the CSV file

    Returns:
        pandas DataFrame
    """
    return pd.read_csv(path)


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

    categorical_columns = [
        'gender',
        'SeniorCitizen',
        'Partner',
        'StreamingTV',
        'PhoneService',
        'InternetService',
        'PaperlessBilling',
        'Churn'
    ]
    numerical_columns = ['MonthlyCharges', 'TotalCharges', 'tenure']

    churn = eda_df.Churn.value_counts()
    plt.figure(figsize=(7, 7))
    plt.pie(churn, labels=['No', 'Yes'], autopct='%1.1f%%')
    plt.title('Presentasi Churn Customer', loc='center', fontsize=18)
    plt.savefig('images/churn_presentation.png')

    _, axis = plt.subplots(1, 3, figsize=(14, 7))
    eda_df[eda_df.Churn == 'No'][numerical_columns].hist(
        bins=20,
        color='blue',
        alpha=0.5,
        ax=axis
    )
    eda_df[eda_df.Churn == 'Yes'][numerical_columns].hist(
        bins=20,
        color='orange',
        alpha=0.5,
        ax=axis
    )
    plt.tight_layout()
    plt.savefig('images/numerical_distribution.png')

    _, axis = plt.subplots(len(categorical_columns), 1, figsize=(10, 35))

    for idx, column in enumerate(categorical_columns):
        sns.countplot(data=eda_df, x=column, hue='Churn', ax=axis[idx])
    plt.tight_layout()
    plt.savefig('images/categorical_distribution.png')

    return eda_df


if __name__ == '__main__':
    customer_df = load_data(path='./data/Telco-Customer-Churn.csv')

    EDA_df = perform_eda(dataframe=customer_df)
