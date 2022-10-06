import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


class Estimator:
    def __init__(self, a_values_base):
        self.a_values_base = a_values_base

    def fit(self, c_values, a_values):
        return self

    def estimate_parameter(self, c_values):
        # assert np.shape(c_values)[1] == np.shape(self.a_values_base)[0], "Shapes c_values and a not coincide."
        pass


class EstimatorNear(Estimator):
    def estimate_parameter(self, c_values):
        super(EstimatorNear, self).estimate_parameter(c_values)
        return self.a_values_base[np.argmax(c_values, axis=1), :]


class EstimatorLinear(Estimator):
    def estimate_parameter(self, c_values):
        super(EstimatorLinear, self).estimate_parameter(c_values)
        return np.einsum("bi,b...->i...", c_values, self.a_values_base)


class EstimatorInv(Estimator):
    def __init__(self, a_values_base):
        super().__init__(a_values_base)
        self.inv_a_values_base = 1.0 / np.array(self.a_values_base)

    def estimate_parameter(self, c_values):
        super(EstimatorInv, self).estimate_parameter(c_values)
        return 1.0 / np.einsum("bi,b...->i...", c_values, self.inv_a_values_base)


# class EstimatorInv(Estimator):
#     def __init__(self, a_values_base):
#         super().__init__(a_values_base)
#         self.inv_a_values_base = [np.linalg.inv(a) for a in self.a_values_base]
#
#     def estimate_parameter(self, c_values):
#         super(EstimatorInv, self).estimate_parameter(c_values)
#         return np.linalg.inv(np.tensordot(c_values, self.inv_a_values_base, axes=((1,), (0,))))


class EstimatorTree(Estimator):
    def __init__(self, a_values_base):
        super().__init__(a_values_base)
        # self.tree = [DecisionTreeRegressor() for _ in range(np.shape(a_values_base)[1])]
        self.tree = [RandomForestRegressor(n_estimators=20, n_jobs=-1) for _ in range(np.shape(a_values_base)[1])]

    def tree_iterator(self, c_values):
        for tree, a_base in zip(self.tree, self.a_values_base.T):
            # X = np.concatenate([c_values, [a_base] * len(c_values)], axis=1)
            X = c_values * np.array([a_base] * len(c_values))
            yield tree, X

    def fit(self, c_values, a_values):
        for i, (tree, X) in enumerate(self.tree_iterator(c_values)):
            tree.fit(X, a_values[:, i])
        return self

    def estimate_parameter(self, c_values):
        super(EstimatorTree, self).estimate_parameter(c_values)
        parameters = []
        for i, (tree, X) in enumerate(self.tree_iterator(c_values)):
            parameters.append(tree.predict(X))
        return np.array(parameters).T


class EstimatorNN(Estimator):
    def __init__(self, a_values_base, hidden_layer_sizes):
        super().__init__(a_values_base)
        # self.tree = [DecisionTreeRegressor() for _ in range(np.shape(a_values_base)[1])]
        self.tree = [MLPRegressor(hidden_layer_sizes=hidden_layer_sizes) for _ in range(np.shape(a_values_base)[1])]

    def tree_iterator(self, c_values):
        for tree, a_base in zip(self.tree, self.a_values_base.T):
            # X = np.concatenate([c_values, [a_base] * len(c_values)], axis=1)
            X = c_values * np.array([a_base] * len(c_values))
            yield tree, X

    def fit(self, c_values, a_values):
        for i, (tree, X) in enumerate(self.tree_iterator(c_values)):
            tree.fit(X, a_values[:, i])
        return self

    def estimate_parameter(self, c_values):
        super(EstimatorNN, self).estimate_parameter(c_values)
        parameters = []
        for i, (tree, X) in enumerate(self.tree_iterator(c_values)):
            parameters.append(tree.predict(X))
        return np.array(parameters).T
