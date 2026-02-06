# scripts/eda/plots.py

"""
EDA visualization functions for churn analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def categorical_distributions(df, feature):
    """
    Plots the distribution of a categorical feature on a given dataframe.

    Parameters:
        df(pd.DataFrame): The input dataframe
        feature: The desired column from the dataframe
    """

    # Plot the distribution
    plt.figure(figsize=(14, 5))
    sns.countplot(x=feature, data=df, palette='deep', order=df[feature].value_counts().index)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def numerical_distribution(df, numerical_features):
    """
    Plots distribution plots with KDE curves for a list of numerical features in the given dataframe

    Parameters:
        df(pd.DataFrame): the input dataframe containing the numerical features
        numerical_features: list of column names to plot
    """

    # Calculate the subplot grid size
    no_of_rows = (len(numerical_features) - 1) // 3 + 1
    no_of_cols = min(3, len(numerical_features))

    # Create subplots
    fig, axes = plt.subplots(nrows=no_of_rows, ncols=no_of_cols, figsize=(16, 4 * no_of_rows))
    axes = axes.flatten() if len(numerical_features) > 1 else [axes]

    # Plot each numerical feature
    for n, feature in enumerate(numerical_features):
        sns.histplot(df[feature], kde=True, ax=axes[n], color='blue', edgecolor='black')
        axes[n].set_title(f"Distribution of {feature}", fontsize=10)
        axes[n].set_xlabel(feature)
        axes[n].set_ylabel('Count')

    # Omit any unused subplots
    for i in range(len(numerical_features), len(axes)):
        fig.delaxes(axes[i])

    # Improve layout spacing
    fig.tight_layout()
    plt.show()


def categorical_churn(df, feature):
    """
    Plots the distribution of a categorical feature, with churn as a comparative variable

    Parameters:
        df(pd.DataFrame): The input dataframe
        feature: The categorical column to investigate
    """

    # Plot the distribution
    plt.figure(figsize=(10, 5))
    churn_count = df.groupby(feature)['Churn'].sum().sort_values(ascending=False)
    top_10_categories = churn_count.head(10).index.tolist()
    sns.countplot(x=feature, hue='Churn', data=df, order=top_10_categories)
    plt.xticks(rotation=45)
    plt.legend(loc='upper right')
    plt.show()


def kde_plots_with_churn(df, feature, type_of_charge):
    """
    Plots the distribution of the numerical features based on the churn rate.

    Parameters:
        df(pd.DataFrame): The input dataframe
        feature: The numerical feature to plot
        type_of_charge: the specific charge type(day, evening, night, international)
    """

    # KDE plots
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x=feature, hue='Churn', fill=True)
    plt.xlabel(f"Total {type_of_charge} Charge")
    plt.ylabel("Density")
    plt.title(f"Churn Distribution by Total {type_of_charge} Charges")
    plt.show()


def correlation_heatmap(df):
    """
    Plots a correlation heatmap that illustrates the correlation between the numerical features and the target(Churn)

    Parameters:
        df(pd.DataFrame): The input dataframe
    """

    # Define plot size
    plt.figure(figsize=(14, 14))

    # Compute the correlation matrix
    corr_matrix = df.corr(numeric_only=True) # Pearson Correlation

    # Create a mask that will hide the upper triangle
    mask = corr_matrix.where(np.tril(np.ones(corr_matrix.shape)).astype(np.bool))

    # Plot the heatmap
    sns.heatmap(
        data=mask,
        cmap='viridis',
        annot=True,
        fmt=".1g",
        vmin=-1
    )

    # Define the title and display plot
    plt.title('Feature Correlatiom Heatmap', fontsize=16)
    plt.show()
