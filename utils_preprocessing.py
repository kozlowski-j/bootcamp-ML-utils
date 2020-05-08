from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

class MyStandardScaler(BaseEstimator, TransformerMixin):
    
    """
    This is a class for Standardization of the features 
    by substructing the mean and scaling to standard deviation.
    
    Attributes: 
        with_mean (bool): Standardize with mean or without it. 
        with_std (bool): Scale with standard deviation or without it. 
    """

    
    def __init__(self,
                 with_mean: bool = True,
                 with_std: bool = True) -> None:
        """ 
        The constructor for MyStandardScaler class. 

        Parameters
        ---------- 
           with_mean (bool): Standardize with mean or without it. 
           with_std (bool): Scale with standard deviation or without it.     
        """

        self.with_mean = with_mean  # save in class attributes with_mean and with_std
        self.with_std = with_std

        self.means = None
        self.stds = None

    def fit(self,
            X: pd.DataFrame,
            y: pd.DataFrame = None) -> object:  # y is ignored
        """ 
        The function to compute the mean and std to be used for later scaling. 

        Parameters
        ----------
            X (pd.DataFrame): The data used to compute the mean and standard deviation used 
                              for later scaling along the features axis. 
            y (pd.DataFrame): ignored
          
        Returns: 
            object MyStandardScaler: Fitted data prepared to be transformed. 
        """
        
        # calculate means and stds for every column
        if self.with_mean:
            self.means = X.mean(axis=0).values

        if self.with_std:
            self.stds = X.std(axis=0).values

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """ 
        The function to perform standardization by centering and scaling 
  
        Parameters
        ---------- 
            X (pd.DataFrame): The data used to scale along the features axis. 
          
        Returns: 
            pd.DataFrame: Scaled data. 
        """
        
        ''' A shallow copy means constructing a new collection object and then populating it with references to the child objects found in the original. 
        The copying process does not recurse and therefore won’t create copies of the child objects themselves. 
        In case of shallow copy, a reference of object is copied in other object. 
        It means that any changes made to a copy of object do reflect in the original object. 
        In python, this is implemented using “copy()” function.'''
        
        X_copied = X.copy()

        # Attention! The orders of the operations matters!
        if self.with_mean:
            X_copied = X_copied - self.means

        if self.with_std:
            X_copied = X_copied / self.stds

        return X_copied


class DropColumns(BaseEstimator, TransformerMixin):
    """
    This is a class for transforming to drop specified columns.
    
    Attributes: 
        columns: str or list
        Names of columns to drop.
    """
    
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
        """
        Transform to drop specified columns.

        Parameters
        ----------
        X (pd.DataFrame): The data used to drop the specified columns. 
        
        Returns: 
            pd.DataFrame: Transformed/truncated data. 
        
        """
        # Drop selected columns.
        X_copied = X.copy()
        X_copied.drop(labels=self.columns_to_drop, axis=1, inplace=True)
        return X_copied



class ColumnsSelectorByType(BaseEstimator, TransformerMixin):
    """
    This is a class for transforming to select columns of specified types.

    Parameters
    ----------
    column_type (str): selected column type. 
    """

    def __init__(self, column_type):
        """
        Transformer to select columns of specified types.

        Parameters
        ----------
        column_type (str): selected column type. 
        """
        
        self.column_type = column_type

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform to select specified columns by type.

        Parameters
        ----------
        X (pd.DataFrame): The data used to select the specified columns. 
        
        Returns: 
            pd.DataFrame: Transformed data with specified columns types. 
        
        """
        return X.select_dtypes(include=self.column_type)


class MissingIndicatorForSparseFeatures(BaseEstimator, TransformerMixin):
    """
    This is a class for implementing a transormer that replaces variables where the percentage 
    of observations contain no data above the threshold, 
    for binary variables with values 1 where the value is given and 0 where there is no values.

    Parameters
    ----------
    threshold (float):  value above that, variables where the percentage of observations containing no data.
    """
    def __init__(self, threshold):
        """
        Missing Indicator for sparse features. Sets threshold.

        Parameters
        ----------
        threshold (float): value above that, variables where the percentage 
        of observations containing no data. 
        """
        if threshold > 1:
            self.threshold = threshold / 100
        else:
            self.threshold = threshold

        # create a container for columns to process
        self.cols_to_transform = None

    def fit(self, X, y=None):
        """
        The function selects column indexes whose percentage of missing values is greater than the threshold.

        Parameters
        ----------
        X (pd.DataFrame): The data used to select the appropriate colum's indexes.  
        
        Returns: 
            object MissingIndicatorForSparseFeatures: Fitted data prepared to be transformed. 
        """
        # select column indexes whose percentage of missing values is greater than the threshold
        column_indicators = X.isnull().mean() > self.threshold

        # write column names to the container
        self.cols_to_transform = X.columns[column_indicators]

        return self

    def transform(self, X):
        """
        The function does a shallow copy and overwrites columns 0 and 1 on this copied dataframe.

        Parameters
        ----------
        X (pd.DataFrame): The data used to overwrite the appropriate columns.  
        
        Returns: 
             pd.DataFrame: Transformed data with overwritten columns. 
        """
        X_copied = X.copy()

        # overwrite columns 0 and 1 on the copied dataframe
        X_copied[self.cols_to_transform] = X_copied[self.cols_to_transform].notnull().astype(int)

        return X_copied


class ReduceRareValues(BaseEstimator, TransformerMixin):
    """
    This is a class for implementing the ReduceRareValues transformer, 
    which reduces the set of nominal variable values by replacing values 
    that are less than the observation threshold with replace_value, defaulting to "rare_value".
    
    Parameters
    ----------
    threshold (float):  value above that, variables where the percentage of observations containing no data.
    raplace_value (str): 'rare_value'
    """

    def __init__(self, threshold, replace_value='rare_value'):
        """
        ReduceRareValues transformer. Sets threshold,replace_value and classes_to_keep.

        Parameters
        ----------
        threshold (float): value above that, variables where the percentage 
        of observations containing no data. 
        raplace_value (str): 'rare_value'
        classes_to_keep (bool): ignored by initiation 
        """
        self.threshold = threshold
        self.replace_value = replace_value
        self.classes_to_keep = None

    def fit(self, X, y=None):

        classes_to_keep_list = list()

        for column in X.columns:
            # check which classes meet the frequency condition
            frequent_classes = X[column].value_counts() > self.threshold

            # get the list of classes we want to keep
            values_to_keep = frequent_classes[frequent_classes == True].index.tolist()

            # add to the list a nested list of pairs column_name, columns_classes_which_we_want_to_preserve
            classes_to_keep_list.append([column, values_to_keep])

        # convert the list into a dictionary
        self.classes_to_keep = dict(classes_to_keep_list)
        return self

    def transform(self, X):
        X_copied = X.copy()

        # for each column, i.e. each item in the dictionary:
        for column, most_freq_values in self.classes_to_keep.items():

            # collect unique values that are not None's from the original column
            unique_values = X_copied[column][X_copied[column].notnull()].unique()

            # select values that are in unique values and which are not in most_freq_values
            values_to_replace = np.setdiff1d(unique_values, most_freq_values)

            # if we have any values to replace, we replace them with the value self.replace_value
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

