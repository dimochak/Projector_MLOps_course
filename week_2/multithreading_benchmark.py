from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import timeit

if __name__ == '__main__':
    X, y = make_classification(
        n_samples=100_000,
        n_classes=2,
        n_features=150,
        n_informative=42
    )
    print('Finished creating dataset')
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = RandomForestClassifier(n_jobs=1)
    print('Start training')
    model.fit(X_train, y_train)
    start = timeit.default_timer()
    y_pred_1 = model.predict(X_test)
    print(f'Time taken for prediction with 1 thread: {timeit.default_timer() - start}')

    model.n_jobs = -1
    start = timeit.default_timer()
    y_pred_2 = model.predict(X_test)
    print(f'Time taken for prediction with many threads: {timeit.default_timer() - start}')

    assert((y_pred_1 == y_pred_2).all())
