from pprint import pprint
from river import feature_selection
from river import stats
from river import stream
from sklearn import datasets

X, y = datasets.make_regression(
    n_samples=100,
    n_features=10,
    n_informative=2,
    random_state=42
)

selector = feature_selection.SelectKBest(
    similarity=stats.PearsonCorr(),
    k=3  # The number of features to keep, defaults to 10
)

for xi, yi, in stream.iter_array(X, y):
    selector = selector.learn_one(xi, yi)

    #pprint(selector.leaderboard)
    pprint(selector.transform_one(xi))