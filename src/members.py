"""Model families of the final ensemble.  Hyperparameters are fixed; they were chosen on the
25 selection folds and confirmed once on 25 fresh folds (see results/SUMMARY.md)."""
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def random_forest(seed):
    """Smooth random forest on the base view: large leaves, 4 candidate features per split."""
    return RandomForestClassifier(n_estimators=1500, min_samples_leaf=8, max_features=4, n_jobs=8, random_state=seed)


def hist_gb():
    """Histogram gradient boosting on the base view."""
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0,
                                          random_state=0)


def logistic():
    """L2 logistic regression on the detail view: median imputation with missing-value indicators and
    standardisation for numbers, one-hot (levels seen at least 20 times) for categories."""
    pre = make_column_transformer(
        (make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler()),
         make_column_selector(dtype_include='number')),
        (OneHotEncoder(handle_unknown='ignore', min_frequency=20), make_column_selector(dtype_include=object)))
    return make_pipeline(pre, LogisticRegression(C=0.03, max_iter=10000))


def catboost(seed):
    """Shallow CatBoost with native categorical handling on the detail view."""
    return CatBoostClassifier(iterations=2000, depth=3, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=seed,
                              thread_count=8, verbose=False, allow_writing_files=False)


def hist_gb_cat():
    """Histogram gradient boosting with native categories on the detail view."""
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0,
                                          min_samples_leaf=20, categorical_features='from_dtype', random_state=0)


def category_levels(av, min_count=10):
    """Label-free: category levels seen at least `min_count` times among the development accounts."""
    return {c: sorted(v for v, n in av[c].value_counts().items() if n >= min_count) for c in av.columns if av[c].dtype == object}


def encode(av, levels):
    """Map every category to a fixed level list (rare or unseen values become 'rare')."""
    out = av.copy()
    for c, keep in levels.items():
        out[c] = pd.Categorical(out[c].where(out[c].isin(keep), 'rare'), categories=keep + ['rare'])
    return out
