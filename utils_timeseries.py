
from copy import deepcopy
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def time_series_cv(model, X, y, scaler_y, folds, train_size_ratio):

    # stworzenie kontenerów na błędy w foldach
    mae_scores = []
    rmse_scores = []
    r2_scores = []

    for train_index, test_index in tscv.split(X):
        if len(train_index) >= train_size_ratio * len(test_index):
            train_index_trunc = train_index[-train_size_ratio * len(test_index):]

            # podział zbioru na trenongowy i testowy
            X_train, X_test = X[train_index_trunc, :], X[test_index, :]
            y_train, y_test = y[train_index_trunc], y[test_index]

            # uczenie kopii modelu i predykcja
            model_tmp = deepcopy(model)
            model_tmp.fit(X_train, y_train)
            y_pred = model_tmp.predict(X_test)

            # reskalowanie do oryginalnych wartości y
            y_pred_r = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
            y_test_r = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

            # wyliczenie miar błędów
            mae_scores.append(mean_absolute_error(y_test_r, y_pred_r))
            rmse_scores.append(mean_squared_error(y_test_r, y_pred_r )* *0.5)
            r2_scores.append(r2_score(y_test_r, y_pred_r))

    return pd.DataFrame(
        {
            'MAE': mae_scores,
            'RMSE': rmse_scores,
            'R2': r2_scores
        }
    )


def get_features_for_prediction(y_pred_series, important_lags, current_timestamp, scaler_x):
    lagged_df = pd.DataFrame({'ZAP': y_pred_series})

    # odtworzenie lagów
    for idx, lag in enumerate(important_lags):
        lagged_df['lag_{}'.format(lag)] = lagged_df['ZAP'].shift(periods=lag - 1)

    # ostatni wiersz ale bez pierwszej kolumny ZAP
    lagged_df = lagged_df.iloc[-1, 1:]

    # zmiene czasowe
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
    # przytnij wektor consumption do takiej długości, żemy można było z niego policzyć lagi
    # długość tego wektora wyznaczamy na podstawie największego istotnego laga dla badanego modelu
    y_pred_df = consumption[:last_timestamp].iloc[-important_lags.max() - 1:, 0]

    # tworzymy obecny timestamp, dla którego będziemy generować predykcję
    current_timestamp = last_timestamp + pd.Timedelta('1H')

    # generujemy autoregresywną predykcję o długośći prediction_horizon
    for i in range(prediction_horizon):
        # pobieramy jeden wiersz zmiennych na podstawie których będziemy robić predykcję dla current_timestamp
        x = get_features_for_prediction(y_pred_df, important_lags, current_timestamp, scaler_x)

        # robimy predykcję i reskalujemy wartości za pomocą scaler_y
        y_pred_one = model.predict(x).ravel()
        y_pred_one = scaler_y.inverse_transform(y_pred_one).ravel()
        print(current_timestamp, y_pred_one[0])

        # dodajemy wypredykowane wartości razem z timestampem do seri predykcji autoregresywnej
        y_pred_df = y_pred_df.append(pd.Series(y_pred_one, index=[current_timestamp]), ignore_index=False)

        # inkrementujemy current_timestamp, dla którego będzie budowana następna predykcja
        current_timestamp += pd.Timedelta('1H')
    return y_pred_df
