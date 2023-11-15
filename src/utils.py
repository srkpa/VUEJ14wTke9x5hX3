import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import get_scorer
from sklearn.utils import all_estimators
from sklearn.base import MetaEstimatorMixin
from sklearn.ensemble._forest import BaseForest
from sklearn.ensemble._weight_boosting import BaseWeightBoosting


def is_meta_estimator(estimator_class: Type[BaseEstimator]) -> bool:
    return issubclass(estimator_class, MetaEstimatorMixin) and not issubclass(
        estimator_class, (BaseForest, BaseWeightBoosting)
    )


def get_model(
    model_name: str = None, task: str = None, **kwargs
) -> List[BaseEstimator]:
    sk_estimators = dict(all_estimators(type_filter=task))
    return (
        [
            model_class()
            for _, model_class in sk_estimators.items()
            if not is_meta_estimator(model_class)
        ]
        if model_name is None
        else [sk_estimators[model_name](**kwargs)]
    )


def train_model(model, X: np.ndarray, y: np.ndarray, scoring: List, **kwargs) -> Tuple:
    start = time.time()
    model = model.fit(X, y)
    train_result = test_model(model, X, y, scoring=scoring)
    stop = time.time()
    duration = stop - start

    return train_result, model, duration


def test_model(
    model: BaseEstimator, X: np.ndarray, y: np.ndarray, scoring: List
) -> Tuple:
    y_pred = model.predict(X)

    results = {}
    # Compute any metric from sklearn
    for name in scoring:
        scorer = get_scorer(name)
        score = scorer._score_func(y_true=y, y_pred=y_pred)
        results[name] = score
        # print(f"Test {name}: {score}")

    predictions = pd.DataFrame(data={"predictions": y_pred, "actual": y})

    return results, predictions


def save_model(fitted_model, filepath: Union[str, Path]) -> None:
    joblib.dump(fitted_model, filepath)
