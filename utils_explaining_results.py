import numpy as np
from typing import Callable


def __sample_space(x: np.ndarray,
                   feature_type: str,
                   sample_resolution: int=100) -> np.ndarray:
    """
    Function samples space according to feature_type.
    Args:
        x: (numpy.ndarray) 1-D array of one feature values
        feature_type: (str) type of feature - either "discrete" or "continuous"
        sample_resolution: (int) size of sampled space
    Returns:
        (numpy.ndarray) vector of sample space.
    """
    if feature_type == 'continuous':
        min_value = x.min()
        max_value = x.max()
        sampled_values = np.linspace(start=min_value, stop=max_value, num=sample_resolution)
    elif feature_type == 'discrete':
        sampled_values = np.unique(x, return_index=False)

    return sampled_values


def plot_pdp(model: object,
             X: np.ndarray,
             feature_number: int,
             feature_name: str,
             feature_type: str = 'continuous') -> pd.DataFrame:
    """
    Function outputs Partial Dependence Plot for given feature number. Implementation based on
    https://christophm.github.io/interpretable-ml-book/pdp.html
    Args:
        model: (object) fitted model with standard predict(X) public method
        X: (numpy.ndarray) multidimensional array of input to the given model
        feature_number: (int) index of feature to compute PDP, where feature vector will be X[:, feature_number]
        feature_name: (str) name of feature
        feature_type: (str) type of feature - either "discrete" or "continuous"
    Returns:
        (pandas.DataFrame) dataframe with one column of PDP values.
    """

    # jeśli zbiór za duży, pobierz 100 losowych przykładów treningowych
    if X.shape[0] > 1000:
        sample_indexes = np.random.randint(X.shape[0], size=1000)
        X = X[sample_indexes, :]

    # dla zmiennej feature_number stwórz siatkę wartości o wybranej rozdzielczości
    sampled_values = __sample_space(x=X[:, feature_number],
                                    feature_type=feature_type,
                                    sample_resolution=100)

    # stwórz zmienną sample_resolution opisującą wielkość stworzonej siatki.
    sample_resolution = sampled_values.shape[0]

    # stwórz pusty kontener na poszerzony zbiór do predykcji
    stacked_instances = np.empty((0, X.shape[1]), float)

    # usuń zmienną, dla której liczone jest PDP i zostaw pomniejszony zbiór
    other_features = np.delete(X,
                               feature_number,
                               axis=1)

    # dla każdej próbki w zbiorze wykonaj:
    for i, row in enumerate(other_features):
        # skopiuj przykład sample_resolution razy
        copied_instances = np.repeat(row.reshape(1, -1),
                                     sample_resolution,
                                     axis=0)

        # wstaw do powtórzonych próbek unikalne wartości zmiennej z wcześniej zbudowanej siatki
        concatenated_instances = np.insert(copied_instances,
                                           feature_number,
                                           sampled_values.ravel(),
                                           axis=1)

        # dodaj rozszerzony zbiór do kontenera
        stacked_instances = np.append(stacked_instances,
                                      concatenated_instances,
                                      axis=0)

    # wykonaj predykcje dla każdej próbki w kontenerze
    y_pred = model.predict(stacked_instances).ravel()

    # stwórz dataframe'a z kolumnami wybranej zmiennej oraz odpowiadającymi predykcjami
    feature_results = pd.DataFrame({feature_name: stacked_instances[:, feature_number],
                                    'output': y_pred})

    # pogrupuj wartości względem wartości wybranej i wylicz z nich średnią
    mean_outputs = feature_results.groupby([feature_name]).mean()

    return mean_outputs


def plot_ice(model: object,
             X: np.ndarray,
             feature_number: int,
             feature_name: str,
             feature_type: str = 'continuous') -> pd.DataFrame:
    """
    Function outputs Individual Conditional Expectation for given feature number. Implementation based on
    https://christophm.github.io/interpretable-ml-book/ice.html
    Args:
        model: (object) fitted model with standard predict(X) public method
        X: (numpy.ndarray) multidimensional array of input to the given model
        feature_number: (int) index of feature to compute ICE, where feature vector will be X[:, feature_number]
        feature_name: (str) name of feature
        feature_type: (str) type of feature - either "discrete" or "continuous"
    Returns:
        (pandas.DataFrame) dataframe with ICE values.
    """

    # jeśli zbiór za duży, pobierz 100 losowych przykładów treningowych
    if X.shape[0] > 1000:
        sample_indexes = np.random.randint(X.shape[0], size=1000)
        X = X[sample_indexes, :]

    # dla zmiennej feature_number stwórz siatkę wartości o wybranej rozdzielczości
    sampled_values = __sample_space(x=X[:, feature_number],
                                    feature_type=feature_type,
                                    sample_resolution=100)

    # stwórz zmienną sample_resolution opisującą wielkość stworzonej siatki.
    sample_resolution = sampled_values.shape[0]

    # stwórz pusty kontener na poszerzony zbiór do predykcji
    stacked_instances = np.empty((0, X.shape[1]), float)

    # usuń zmienną, dla której liczone jest PDP i zostaw pomniejszony zbiór
    other_features = np.delete(X,
                               feature_number,
                               axis=1)

    # stwórz pustą lista, która będzie przechowywać informacje, z której próbki pochodzi który wiersz w finalnym zbiorze
    row_indicator = []

    # dla każdej próbki w zbiorze wykonaj:
    for i, row in enumerate(other_features):
        # skopiuj przykład sample_resolution razy
        copied_instances = np.repeat(row.reshape(1, -1),
                                     sample_resolution,
                                     axis=0)

        # wstaw do powtórzonych próbek unikalne wartości zmiennej z wcześniej zbudowanej siatki
        concatenated_instances = np.insert(copied_instances,
                                           feature_number,
                                           sampled_values.ravel(),
                                           axis=1)

        # dodaj rozszerzony zbiór do kontenera
        stacked_instances = np.append(stacked_instances,
                                      concatenated_instances,
                                      axis=0)

        # dołącz do listy wektor odpowiadający indeksowi przewarzanej próbki w pętli
        row_indicator += ((np.ones(sample_resolution) * i).ravel().tolist())

    # wykonaj predykcje dla każdej próbki w kontenerze
    y_pred = model.predict(stacked_instances).ravel()

    # stwórz dataframe'a z kolumnami wybranej zmiennej, identyfikatorami oryginalnych próbek oraz odpowiadającymi predykcjami
    feature_results = pd.DataFrame({feature_name: stacked_instances[:, feature_number],
                                    'output': y_pred,
                                    'sample_id': row_indicator})

    # stwórz dataframe'a z kolumną opisującą siatkę wartości badanej zmiennej
    samples_groups_df = pd.DataFrame(
        {feature_name: feature_results[feature_results['sample_id'] == 0][feature_name].values})

    # ustaw stworzoną kolumnę jako indeks
    samples_groups_df.set_index(feature_name, drop=True, inplace=True)

    # dla każdej z oryginalnych próbek:
    for slice_num in range(X.shape[0]):
        # dodaj kolumnę opisująca uzyskane wyniki predykcji, znajdując odpowiednie sample_id
        samples_groups_df[str(slice_num)] = feature_results[feature_results['sample_id'] == slice_num]['output'].values

    return samples_groups_df


def permutation_importance(model: object,
                           X: np.ndarray,
                           y: np.ndarray,
                           feature_number: int,
                           error_func: Callable,
                           samples_numb: int = 1000) -> np.ndarray:
    """
    Function outputs permutation importance for given feature number. Implementation based on
    https://christophm.github.io/interpretable-ml-book/feature-importance.html
    Args:
        model: (object) fitted model with standard predict(X) public method
        X: (numpy.ndarray) multidimensional array of input to the given model
        y: (numpy.ndarray) array of outputs matched to X matrix
        feature_number: (int) index of feature to compute permutation importance, where feature vector will be X[:, feature_number]
        error_func: (function) pointer to error function, based on which error will be computed
        samples_numb: (int) number of permutation trials
    Returns:
        (numpy.ndarray) vector of increased errors of samples_numb length (0 - 0%, 1=100%).
    """
    # wylicz predykcje dla zadanego zbioru
    y_pred = model.predict(X).ravel()

    # dla wyliczonych predykcji oblicz błąd zgodnie z podaną metodą
    error = error_func(y.ravel(), y_pred.ravel())

    # stwórz pustą listę na błędy predykcji
    scores = []

    # powtórz samples_numb-krotnie:
    for sample in range(samples_numb):
        # skopiuj zbiór danych, zapobiegając jego nadpisaniu
        shuff_test = X.copy()

        # permutuj wartości w badanej zmiennej
        shuff_test[:, feature_number] = np.random.permutation(shuff_test[:, feature_number])

        # wyznacz błąd predykcji na tak upośledzonym zbiorze i doadj go do listy
        score = error_func(y.ravel(), model.predict(shuff_test).ravel())
        scores.append(score)

    # wyznacz względne błędy porównując je do oryginalnej wartości
    scores = np.asarray(scores)
    importances = (scores - error) / error

    return importances.ravel()
