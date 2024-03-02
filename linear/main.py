# обратная матрица пенроуза --------------------------------------------------------------------------------------------

def _pseudoinverse_matrix(self, matrix: np.ndarray) -> np.ndarray:
    """
    The pseudoinverse (Φ^+) of the design matrix Φ can be computed using the formula:

            Ф = U * Σ * V^T
            Φ^+ = V * Σ^+ * U^T

            Where:
            - U, Σ, and V are the matrices resulting from the SVD of Φ.

            The Σ^+ is computed as:

            Σ'_{i,j} =
            | 1/Σ_{i,j}, if Σ_{i,j} > ε * max(N, M+1) * max(Σ)
            | 0, otherwise

            and then:
            Σ^+ = Σ'^T

            where:
            - ε is the machine epsilon, which can be obtained in Python using:
                ε = sys.float_info.epsilon
            - N is the number of rows in the design matrix.
            - M is the number of base functions (without φ_0(x_i)=1).

    """

    svd = np.linalg.svd(matrix, full_matrices=True)

    n = matrix.shape[0]
    m = matrix.shape[1]

    s_pseudo_inv = [
        svd.S[i] / (svd.S[i] * svd.S[i] + self.regularization_rate) if svd.S[i] > np.finfo(float).eps * max(n,
                                                                                                            m + 1) * matrix.max()
        else 0 for i in range(len(svd.S))]

    # construct diag matrix from list
    s_pseudo_inv = np.diag(s_pseudo_inv)

    # do not affect initial weight
    s_pseudo_inv[0, 0] = 1.0 / svd.S[0]

    return svd.Vh.T @ s_pseudo_inv @ svd.U.T


# normal equation ------------------------------------------------------------------------------------------------------

def _calculate_weights_normal_equation(self, pseudoinverse_plan_matrix: np.ndarray, targets: np.ndarray) -> None:
    """
        The weights (w) can be computed using the formula:
        w = Φ^+ * t

        Where:
        - Φ^+ is the pseudoinverse of the design matrix
        - t is the target vector.
    """

    self.weights = pseudoinverse_plan_matrix @ targets


# пересчет весов при градиентном спуске --------------------------------------------------------------------------------

def _calculate_gradient(self, plan_matrix: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    The gradient of the error with respect to the weights (∆w E) can be computed using the formula:
    ∆w E = (2/N) * Φ^T * (Φ * w - t)

    Where:
    - Φ is the design matrix.
    - w is the weight vector.
    - t is the vector of target values.
    - N is the number of data points.

    This formula represents the partial derivative of the mean squared error with respect to the weights.

    For regularisation: ∆w E = (2/N) * Φ^T * (Φ * w - t)  + λ * w
    """

    return (2 / len(targets)) * plan_matrix.T @ (
            np.dot(plan_matrix, self.weights) - targets) + self.regularization_rate * self.weights


def _calculate_weights_gradient(self, plan_matrix: np.ndarray, targets: np.ndarray) -> None:
    """
    At each iteration of gradient descent, the weights are updated using the formula:
    w_{k+1} = w_k - γ * ∇_w E(w_k)

    Where:
    - w_k is the current weight vector at iteration k.
    - γ is the learning rate, determining the step size in the direction of the negative gradient.
    - ∇_w E(w_k) is the gradient of the cost function E with respect to the weights w at iteration k.

    This iterative process aims to find the weights that minimize the cost function E(w).
"""

    grad = self._calculate_gradient(plan_matrix, targets)
    self.weights = self.weights - self.learning_rate * grad


# предсказание ---------------------------------------------------------------------------------------------------------

def __call__(self, inputs: np.ndarray) -> np.ndarray:
    plan_matrix = self._plan_matrix(inputs)
    predictions = self.calculate_model_prediction(plan_matrix)

    return predictions  # return prediction of the model


# подсчет ошибки -------------------------------------------------------------------------------------------------------

def MSE(predictions: np.ndarray, targets: np.ndarray) -> float:
    return (1 / len(targets)) * np.sum(np.power(targets - predictions, 2))


# базовые функции ------------------------------------------------------------------------------------------------------

m = 100
base_functions = [lambda x, i=i: np.power(x, i) for i in range(m)]
