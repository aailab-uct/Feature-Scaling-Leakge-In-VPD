from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from scipy.stats import uniform, loguniform, randint


def set_param_grid(n_features, random_state=42):
    param_grid = {
                    "svm": {"estimator": SVC(max_iter=int(1e5), kernel="rbf", random_state=random_state),
                            "param_space": dict(model__C = [0.1, 1, 10, 100, 1000],
                                                model__gamma = [1e-3, 1e-2, 1e-1]),
                            },
                    "knn": {"estimator": KNeighborsClassifier(),
                            "param_space": dict(model__n_neighbors = list(range(1, 51, 2)),
                                                model__weights = ['uniform', 'distance']),
                            },
                    "gaussianNB": {"estimator": GaussianNB(),
                                   "param_space": dict(model__var_smoothing = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]),
                                   },
                    # "gaussian_process": {"estimator": GaussianProcessClassifier(n_restarts_optimizer=0,
                    #                                                             max_iter_predict=10,
                    #                                                             random_state=random_state),
                    #                      "param_space": dict(model__kernel=list(x*RBF(l) for x in
                    #                                                             loguniform.rvs(0.01, 100, size=100,
                    #                                                                                   random_state=random_state)
                    #                                                               for l in loguniform.rvs(0.01, 100, size=100,
                    #                                                                                   random_state=random_state)
                    #                                                      )
                    #                                          ),
                    #                      "iters":100
                    #                      },
                    "lda": {"estimator": LinearDiscriminantAnalysis(),
                            "param_space": dict(model__tol=[1e-5, 1e-4, 1e-3, 1e-2]),
                            },
                    "qda": {"estimator": QuadraticDiscriminantAnalysis(),
                            "param_space": dict(model__reg_param=[0, 0.2, 0.4, 0.6, 0.8, 1],
                                                model__tol=[1e-5, 1e-4, 1e-3, 1e-2])
                            },
                    "dt": {"estimator": DecisionTreeClassifier(random_state=random_state),
                           "param_space": dict(model__splitter=["best", "random"],
                                               model__min_samples_leaf=list(range(1, 21, 2)),
                                               model__max_features=[0.1, 0.3, 0.5, 0.7, 0.9, 1]),
                           },
                    "rf": {"estimator": RandomForestClassifier(random_state=random_state),
                           "param_space": dict(model__min_samples_leaf=list(range(1, 21, 2)),
                                               model__max_features=[0.1, 0.3, 0.5, 0.7, 0.9, 1],
                                               model__n_estimators=[10, 20, 30, 40]),
                           },
                    "adaboost": {"estimator": AdaBoostClassifier(random_state=random_state),
                                 "param_space": dict(model__n_estimators=[10, 20, 30, 40],
                                                     model__learning_rate=[0.1, 0.5,  1., 5, 10]),
                                 },
                    "mlp": {"estimator": MLPClassifier(random_state=random_state, solver="lbfgs", max_iter=100),
                            "param_space": dict(model__alpha=[1e-5, 1e-4, 1e-3],
                                                model__hidden_layer_sizes=[int(size * n_features) for size in [1.6, 1.8, 2]] + \
                                                                   [[int(size * n_features),
                                                                     int(size * n_features / 2)]
                                                                    for size in [1.6, 1.8, 2]]),
                            }
                }
    return param_grid