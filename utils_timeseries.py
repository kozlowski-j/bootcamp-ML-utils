import pandas as pd
from copy import deepcopy
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import adfuller


def time_series_cv(model: object, X: pd.DataFrame, y: np.ndarray,
                   scaler_y: object, folds: int, train_size_ratio: float):
    """
    Function performing cross-validation on time series data based on sklearn,
    with modified strategy: train set is adjusted to be of the same size in each fold.

    Parameters
    ----------
    model: object
        Sklearn model with predict method.
    X : pandas.DataFrame
        Set of features.
    y : array_like
        Target variable.
    scaler_y: object
        Scaling transformer used to transform y.
    folds: int
        Number of folds.
    train_size_ratio: float
        Should be between 0.0 and 1.0.

    Returns
    -------
    metrics: pandas.DataFrame
        Metrics per each split.
    """

    # Create containers for metrics.
    mae_scores = []
    rmse_scores = []
    r2_scores = []

    # Start transformer.
    tscv = TimeSeriesSplit(folds)

    # For each fold of TimeSeriesSplit truncate training set.
    for train_index, test_index in tscv.split(X):

        # Train size based on given ratio parameter.
        train_set_size = train_size_ratio * len(test_index)

        # Assure that training set is big enough.
        if len(train_index) >= train_set_size:

            # Truncate training set index.
            train_index_trunc = train_index[-train_set_size:]

            # Split into train and test sets.
            X_train, X_test = X[train_index_trunc, :], X[test_index, :]
            y_train, y_test = y[train_index_trunc], y[test_index]

            # Train model and predict values.
            model_tmp = deepcopy(model)
            model_tmp.fit(X_train, y_train)
            y_pred = model_tmp.predict(X_test)

            # Inverse scaling to original values.
            y_pred_r = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
            y_test_r = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

            # Get result's metrics.
            mae_scores.append(mean_absolute_error(y_test_r, y_pred_r))
            rmse_scores.append(mean_squared_error(y_test_r, y_pred_r)**0.5)
            r2_scores.append(r2_score(y_test_r, y_pred_r))

    # TODO: redo the output to return also the model and data
    return pd.DataFrame(
        {
            'MAE': mae_scores,
            'RMSE': rmse_scores,
            'R2': r2_scores
        }
    )


def get_features_for_prediction(y_pred_series, important_lags, current_timestamp, scaler_x):
    """
    Helper function for auto-regression.

    Parameters
    ----------
    y_pred_series
    important_lags
    current_timestamp
    scaler_x

    Returns
    -------

    """
    lagged_df = pd.DataFrame({'ZAP': y_pred_series})

    # odtworzenie lagów
    for idx, lag in enumerate(important_lags):
        lagged_df['lag_{}'.format(lag)] = lagged_df['ZAP'].shift(periods=lag - 1)

    # ostatni wiersz ale bez pierwszej kolumny ZAP
    lagged_df = lagged_df.iloc[-1, 1:]

    # zmiene czasowe
    # TODO: get this functions
    lagged_df['is_weekend'] = is_weekend(current_timestamp)
    lagged_df['HoD_sin'] = get_hour_sin(current_timestamp)
    lagged_df['HoD_cos'] = get_hour_cos(current_timestamp)
    lagged_df['MoY_sin'] = get_month_sin(current_timestamp)
    lagged_df['MoY_cos'] = get_month_cos(current_timestamp)
    #     print(lagged_df)

    # skalowanie danych
    x = lagged_df.values.reshape(1, -1)
    x_scaled = scaler_x.transform(x)
    return x_scaled


def predict_autoregressive(model, consumption, important_lags, prediction_horizon, last_timestamp, scaler_x, scaler_y):
    """

    Parameters
    ----------
    model: object
        Sklearn model with predict method.
    consumption: pandas.DataFrame
        Features table.
    important_lags:
    prediction_horizon:
    last_timestamp:
    scaler_x: object
        Scaling transformer object used to scale X having inverse_transform method.
    scaler_y: object
        Scaling transformer object used to scale y having inverse_transform method.

    Returns
    -------
    y_pred_df: pandas.DataFrame


    """
    # TODO: Run this function and check if rly works.
    # TODO: divide it into more steps
    # przytnij wektor consumption do takiej długości, żemy można było z niego policzyć lagi
    # długość tego wektora wyznaczamy na podstawie największego istotnego laga dla badanego modelu
    y_pred_df = consumption[:last_timestamp].iloc[-important_lags.max() - 1:, 0]

    # Get current timestamp to generate prediction for it.
    current_timestamp = last_timestamp + pd.Timedelta('1H')

    # Create auto-regressive prediction of length prediction_horizon.
    # generujemy autoregresywną predykcję o długośći prediction_horizon
    for i in range(prediction_horizon):
        # pobieramy jeden wiersz zmiennych na podstawie których będziemy robić predykcję dla current_timestamp
        x = get_features_for_prediction(y_pred_df, important_lags, current_timestamp, scaler_x)

        # Predict y and inverse scale result.
        y_pred_one = model.predict(x).ravel()
        y_pred_one = scaler_y.inverse_transform(y_pred_one).ravel()
        print(current_timestamp, y_pred_one[0])

        # dodajemy wypredykowane wartości razem z timestampem do seri predykcji autoregresywnej
        # Update result DataFrame.
        y_pred_df = y_pred_df.append(pd.Series(y_pred_one, index=[current_timestamp]), ignore_index=False)

        # Prepare timestamp for next prediction.
        current_timestamp += pd.Timedelta('1H')
    return y_pred_df


def stationary(df_test):
    if df_test[1] < 0.05:
        print('Time series is stationary.')
    else:
        print('Time series is not stationary')


def check_time_series_stationary(y, rolling_len=12):
    """
    Perform Dickey-Fuller test for stationarity. Plot TS and autocorrelation.

    Parameters
    ----------
    y: array-like
        Time Series object.
    rolling_len: int
        Rolling mean window width.

    Returns
    -------

    """
    y = pd.Series(y)
    rolling_mean = y.rolling(rolling_len).mean()
    rolling_var = y.rolling(rolling_len).var()

    f, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].plot(y)
    ax[0].plot(rolling_mean, label='Rolling mean')
    ax[0].plot(rolling_var, label='Rolling var')
    ax[0].legend()

    pd.plotting.autocorrelation_plot(y, ax=ax[1])

    adfuller_stats = adfuller(y)
    print(f'Adfuller statistic: {np.round(adfuller_stats[0], 4)}')
    print(f'Adfuller p-value: {np.round(adfuller_stats[1], 4)}')
    stationary(adfuller_stats)

