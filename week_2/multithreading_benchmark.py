import numpy as np
from joblib import Parallel, delayed
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import timeit
from tqdm import tqdm

if __name__ == '__main__':
    X, y = make_classification(
        n_samples=5_000,
        n_classes=2,
        n_features=150,
        n_informative=42
    )
    print('Finished creating dataset')
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = RandomForestClassifier(n_jobs=-1)
    print('Start training')
    model.fit(X_train, y_train)

    start = timeit.default_timer()
    y_pred_1 = []
    for test_example in tqdm(X_test):
        y_pred_1.append(model.predict(np.array(test_example).reshape(1, -1)))
    print(f'Time taken for prediction with 1 thread: {timeit.default_timer() - start}')

    start = timeit.default_timer()
    y_pred_2 = Parallel(n_jobs=-1, backend='threading')(
        delayed(model.predict)(np.array(test_example).reshape(1, -1)) for test_example in X_test)
    print(f'Time taken for prediction with many threads (threading backend): {timeit.default_timer() - start}')
    assert (y_pred_1 == y_pred_2)

    start = timeit.default_timer()
    y_pred_3 = Parallel(n_jobs=-1, backend='multiprocessing')(
        delayed(model.predict)(np.array(test_example).reshape(1, -1)) for test_example in X_test)
    print(f'Time taken for prediction with many threads (multiprocessing): {timeit.default_timer() - start}')
    assert (y_pred_2 == y_pred_3)

    start = timeit.default_timer()
    y_pred_4 = Parallel(n_jobs=-1, backend='loky')(
        delayed(model.predict)(np.array(test_example).reshape(1, -1)) for test_example in X_test)
    print(f'Time taken for prediction with many threads (loky): {timeit.default_timer() - start}')
    assert (y_pred_3 == y_pred_4)
