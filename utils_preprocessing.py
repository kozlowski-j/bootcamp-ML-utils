from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np


class MyStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self,
                 with_mean: bool = True,
                 with_std: bool = True) -> None:

        self.with_mean = with_mean  # przechować w polach klasy with_mean i with_std
        self.with_std = with_std

        self.means = None
        self.stds = None

    def fit(self,
            X: pd.DataFrame,
            y: pd.DataFrame = None) -> object:  # nie korzystamy z y
        # dla każdej z kolumn liczymy średnie i odchylenia standardowe

        if self.with_mean:
            self.means = X.mean(axis=0).values

        if self.with_std:
            self.stds = X.std(axis=0).values

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # przekształcamy X i zwracamy X_scaled
        X_copied = X.copy()

        # uwaga! Kolejność  operacji ma znaczenie!
        if self.with_mean:
            X_copied = X_copied - self.means

        if self.with_std:
            X_copied = X_copied / self.stds

        return X_copied


class DropColumns(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        """
        Transformer to drop specified columns.

        Parameters
        ----------
        columns: str or list
            Names of columns to drop.
        """
        # Keep columns' names.
        self.columns_to_drop = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Drop selected columns.
        X_copied = X.copy()
        X_copied.drop(labels=self.columns_to_drop, axis=1, inplace=True)
        return X_copied


class ColumnsSelectorByType(BaseEstimator, TransformerMixin):
    """
    Transformer to select columns of specified types.
    """

    def __init__(self, column_type):
        self.column_type = column_type

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.select_dtypes(include=self.column_type)


class MissingIndicatorForSparseFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        if threshold > 1:
            self.threshold = threshold / 100
        else:
            self.threshold = threshold

        # stwórz kontener na kolumny do przetworzenia
        self.cols_to_transform = None

    def fit(self, X, y=None):
        # wybierz indeksy kolumn, których odeset brakujących wartości jest większy niż threshold
        column_indicators = X.isnull().mean() > self.threshold

        # zapisz nazwy kolumn do konetera
        self.cols_to_transform = X.columns[column_indicators]

        return self

    def transform(self, X):
        X_copied = X.copy()

        # na skopoiwanym dataframe nadpisz kolumny 0 i 1
        X_copied[self.cols_to_transform] = X_copied[self.cols_to_transform].notnull().astype(int)

        return X_copied


class ReduceRareValues(BaseEstimator, TransformerMixin):

    def __init__(self, threshold, replace_value='rare_value'):
        self.threshold = threshold
        self.replace_value = replace_value
        self.classes_to_keep = None

    def fit(self, X, y=None):

        classes_to_keep_list = list()

        for column in X.columns:
            # sprawdzamy, które klasy pełniają warunek częstości
            frequent_classes = X[column].value_counts() > self.threshold

            # pozyskujemy listę klas, kóre chcemy zachować
            values_to_keep = frequent_classes[frequent_classes == True].index.tolist()

            # dodajemy do listy zagnieżdzoną listę par nazwa_kolumny, klasy_kolumny_które_chcemy_zachować
            classes_to_keep_list.append([column, values_to_keep])

        # konwertujemy listę na słownik
        self.classes_to_keep = dict(classes_to_keep_list)
        return self

    def transform(self, X):
        X_copied = X.copy()

        # dla każdej kolumny, czyli każdego itema w słowniku:
        for column, most_freq_values in self.classes_to_keep.items():

            # zbierz unikalne wartości, które nie są None'ami, z oryginalnej kolumny
            unique_values = X_copied[column][X_copied[column].notnull()].unique()

            # wybierz wartości, które znajdując się w unique_values a kórych nie ma w most_freq_values
            values_to_replace = np.setdiff1d(unique_values, most_freq_values)

            # jeśli mamy jakieś wartośi do zastąpienia, zastępujemy je wartością self.replace_value
            if len(values_to_replace) > 0:
                X_copied[column].replace(values_to_replace, self.replace_value, inplace=True)

        return X_copied


class SimpleImputerWrapper(BaseEstimator, TransformerMixin):

    def __init__(self, strategy, fill_value=None):
        self.imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        self.columns = None

    def fit(self, X, y=None):
        self.imputer.fit(X)
        self.columns = X.columns
        return self

    def transform(self, X):
        return pd.DataFrame(self.imputer.transform(X), columns=self.columns)