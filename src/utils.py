import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper
from sklearn.linear_model import LinearRegression


matplotlib.rcParams['figure.figsize'] = (8, 4)
sb.set(font_scale=1.)


def calculate_residuals(model, features, labels):
    '''Calculates residuals between true value `labels` and predicted value.'''
    y_pred = model.predict(features)
    df_results = pd.DataFrame({'Actual': labels, 'Predicted': y_pred})
    df_results['Residuals'] = abs(df_results['Actual']) - abs(df_results['Predicted'])
    return df_results


def linear_assumption(model, features, labels, p_value_thresh=0.05, plot=True):
    '''
    Linear assumption: assumes linear relation between the independent and dependent variables.
    - Za statsmodels model: koristi F-test (f_pvalue)
    - Za sklearn model: fituje statsmodels OLS interno i koristi F-test

    Returns:
    - is_linearity_found: bool
    - p_value: float
    '''
    df_results = calculate_residuals(model, features, labels)
    y_pred = df_results['Predicted']

    if plot:
        plt.figure(figsize=(6, 6))
        plt.scatter(labels, y_pred, alpha=.5)
        line_coords = np.linspace(np.concatenate([labels, y_pred]).min(), np.concatenate([labels, y_pred]).max())
        plt.plot(line_coords, line_coords, color='darkorange', linestyle='--')
        plt.title('Linear assumption')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.show()

    if isinstance(model, RegressionResultsWrapper):
        p_value = model.f_pvalue
    elif isinstance(model, LinearRegression):
        features_sm = sm.add_constant(features)
        lr_sm = sm.OLS(labels, features_sm).fit()
        p_value = lr_sm.f_pvalue
    else:
        raise ValueError("Model mora biti sklearn LinearRegression ili statsmodels RegressionResultsWrapper")

    is_linearity_found = p_value < p_value_thresh
    return is_linearity_found, p_value


def independence_of_errors_assumption(model, features, labels, plot=True):
    '''
    Independence of errors: assumes independent errors.
    Testing autocorrelation using Durbin-Watson Test.

    Interpretation of `d` value:
    - 1.5 <= d <= 2: No autocorrelation (independent residuals).
    - d < 1.5: Positive autocorrelation.
    - d > 2: Negative autocorrelation.

    Returns:
    - autocorrelation: 'positive', 'negative', or None
    - dw_value: float
    '''
    df_results = calculate_residuals(model, features, labels)

    if plot:
        sb.scatterplot(x='Predicted', y='Residuals', data=df_results)
        plt.axhline(y=0, color='darkorange', linestyle='--')
        plt.show()

    from statsmodels.stats.stattools import durbin_watson
    dw_value = durbin_watson(df_results['Residuals'])

    if dw_value < 1.5:
        autocorrelation = 'positive'
    elif dw_value > 2:
        autocorrelation = 'negative'
    else:
        autocorrelation = None

    return autocorrelation, dw_value


def normality_of_errors_assumption(model, features, labels, p_value_thresh=0.05, plot=True):
    '''
    Normality of errors: assumes normally distributed residuals around zero.
    Testing using the Anderson-Darling test.

    Returns:
    - dist_type: 'normal' or 'non-normal'
    - p_value: float
    '''
    df_results = calculate_residuals(model, features, labels)

    if plot:
        plt.title('Distribution of residuals')
        sb.histplot(df_results['Residuals'], kde=True, kde_kws={'cut': 3})
        plt.show()

    from statsmodels.stats.diagnostic import normal_ad
    p_value = normal_ad(df_results['Residuals'])[1]
    dist_type = 'normal' if p_value >= p_value_thresh else 'non-normal'
    return dist_type, p_value


def equal_variance_assumption(model, features, labels, p_value_thresh=0.05, plot=True):
    '''
    Equal variance: assumes that residuals have equal variance across the regression line.
    Testing using Goldfeld-Quandt test.

    Returns:
    - dist_type: 'equal' or 'non-equal'
    - p_value: float
    '''
    df_results = calculate_residuals(model, features, labels)

    if plot:
        sb.scatterplot(x='Predicted', y='Residuals', data=df_results)
        plt.axhline(y=0, color='darkorange', linestyle='--')
        plt.show()

    features_sm = sm.add_constant(features) if isinstance(model, LinearRegression) else features
    p_value = sm.stats.het_goldfeldquandt(df_results['Residuals'], features_sm)[1]
    dist_type = 'equal' if p_value >= p_value_thresh else 'non-equal'
    return dist_type, p_value


def perfect_collinearity_assumption(features: pd.DataFrame, plot=True):
    '''
    Perfect collinearity: assumes no perfect correlation between two or more features.

    Returns:
    - has_perfect_collinearity: bool
    '''
    correlation_matrix = features.corr()

    if plot:
        sb.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.1)
        plt.title('Matrica korelacije')
        plt.show()

    np.fill_diagonal(correlation_matrix.values, np.nan)
    pos_perfect_collinearity = (correlation_matrix > 0.999).any().any()
    neg_perfect_collinearity = (correlation_matrix < -0.999).any().any()
    has_perfect_collinearity = pos_perfect_collinearity or neg_perfect_collinearity
    return has_perfect_collinearity


def are_assumptions_satisfied(model, features, labels, p_value_thresh=0.05):
    '''Check if all assumptions in multiple linear regression are satisfied.'''
    is_linearity_found, p_val_l = linear_assumption(model, features, labels, p_value_thresh, plot=False)
    autocorrelation, dw_value = independence_of_errors_assumption(model, features, labels, plot=False)
    n_dist_type, p_val_n = normality_of_errors_assumption(model, features, labels, p_value_thresh, plot=False)
    e_dist_type, p_val_e = equal_variance_assumption(model, features, labels, p_value_thresh, plot=False)
    has_perfect_collinearity = perfect_collinearity_assumption(features, plot=False)

    print(f"L — Linearnost:        {'PASSED' if is_linearity_found else 'FAILED'} (p={p_val_l:.4f})")
    print(f"I — Nezavisnost:       {'PASSED' if autocorrelation is None else 'FAILED'} (DW={dw_value:.4f}, autokorelacija={autocorrelation})")
    print(f"N — Normalnost:        {'PASSED' if n_dist_type == 'normal' else 'FAILED'} (p={p_val_n:.4f}, {n_dist_type})")
    print(f"E — Jednaka varijansa: {'PASSED' if e_dist_type == 'equal' else 'FAILED'} (p={p_val_e:.4f}, {e_dist_type})")
    print(f"Kolinearnost:          {'FAILED — postoji' if has_perfect_collinearity else 'PASSED — nema savrsene kolinearnosti'}")

    all_satisfied = (
        is_linearity_found and
        autocorrelation is None and
        n_dist_type == 'normal' and
        e_dist_type == 'equal' and
        not has_perfect_collinearity
    )
    print(f"\nSve pretpostavke ispunjene: {'DA' if all_satisfied else 'NE'}")
    return all_satisfied