import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import numpy as np
import mltable
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller


def read_delta_table_in_datalake(path_on_datastore_, datastore_name_, subscription_, resource_group_, workspace_, rt_uri=False, **kwargs):
    uri = f'azureml://subscriptions/{subscription_}/resourcegroups/{resource_group_}/workspaces/{workspace_}/datastores/{datastore_name_}/paths/{path_on_datastore_}'
    if rt_uri: print("uri : ", uri)
    tbl = mltable.from_delta_lake(delta_table_uri=uri, **kwargs)
    return tbl.to_pandas_dataframe()


def debug_plot(df):
    display(df.head(5)) # type: ignore
    display(df.info()) # type: ignore
    msno.matrix(df)
    plt.show()


def lineplot_large(df, figsize=(16, 5)):
    plt.figure(figsize=figsize)
    sns.set(style="whitegrid")
    sns.lineplot(df)
    plt.show()


def adfuller_test(series, signif=0.05, name='', verbose=False):
    # Main compute
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue']

    # Users informations
    if verbose:
        print(f'\nAugmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
        print(f'Null Hypothesis: Data has unit root. Non-Stationary.')
        print(f'Significance Level = {signif}')
        print(f'Test Statistic = {output["test_statistic"]}')
        print(f'No. Lags Chosen = {output["n_lags"]}')

        for key,val in r[4].items():
            print(f' Critical value {str(key).ljust(6)} = {round(val, 3)}')
            if p_value <= signif:
                print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
                print(f" => Series is Stationary.")
            else:
                print(f" => P-Value = {p_value}. Weak evidence to reject the   Null Hypothesis.")
                print(f" => Series is Non-Stationary.")
    return p_value


def granger_test(target_series, reference_series, max_lags=5, verbose=False):
    # Checking
    assert target_series.shape == reference_series.shape, "Both series must have the same shape." # shape
    for name, series in {"target_series":target_series, "reference_series":reference_series}.items():
        assert series.isna().any() == False, f"Nan value(s) fouds in {name}." # NaN
        assert series.dtypes in ['int', 'float'], f"{name} dtypes is not int/float." # dtypes
        pval = adfuller_test(series, signif=0.05, name=name, verbose=False) # stationarity
        assert pval > 0.05, f"ERROR: {name} must be stationary (ADF test p-value = {pval})."
   
    # Granger test
    ts_df = pd.DataFrame({'ts1':target_series, 'ts2':reference_series})
    gtest = grangercausalitytests(ts_df, 3, verbose=verbose)

    return gtest