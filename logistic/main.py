# раздел датасета ------------------------------------------------------------------------------------------------------
def _divide_into_sets(self):
    inputs_length = len(self.inputs)
    indexes = np.random.permutation(inputs_length)

    train_set_size = int(self.train_set_percent * inputs_length)
    valid_set_size = int(self.valid_set_percent * inputs_length)

    randomized_inputs = self.inputs[indexes]
    randomized_targets = self.targets[indexes]

    self.inputs_train = randomized_inputs[:train_set_size]
    self.inputs_valid = randomized_inputs[train_set_size:train_set_size + valid_set_size]
    self.inputs_test = randomized_inputs[train_set_size + valid_set_size:]

    self.targets_train = randomized_targets[:train_set_size]
    self.targets_valid = randomized_targets[train_set_size:train_set_size + valid_set_size]
    self.targets_test = randomized_targets[train_set_size + valid_set_size:]


# стандартизация -------------------------------------------------------------------------------------------------------

def standartization(self):
    # means - массив средних значений, заполняется при чтении датасета means.append(float(numpy.mean(self.inputs_train[i])))
    # stds - массив стандартных значений: stds.append(float(numpy.std(self.inputs_train[i])))

    standart = lambda vec: np.array([(vec[i] - self.means[i]) / self.stds[i] for i in range(len(vec))])

    self.inputs_train = np.apply_along_axis(standart, 1, self.inputs_train)
    self.inputs_valid = np.apply_along_axis(standart, 1, self.inputs_valid)
    self.inputs_test = np.apply_along_axis(standart, 1, self.inputs_test)


# ван хот энкодинг вектор ----------------------------------------------------------------------------------------------

def onehotencoding(targets: np.ndarray, number_of_classes: int) -> np.ndarray:
    matr = np.zeros((len(targets), number_of_classes))
    matr[np.arange(len(targets)), targets] = 1

    return matr

# целевая функция ------------------------------------------------------------------------------------------------------

def __target_function(self, inputs: np.ndarray, targets: np.ndarray,
                      z: np.ndarray) -> float:
    """
    - Σ(i=0 to N-1) Σ(k=0 to K-1) t_ik * (ln(Σ(l=0 to K-1) e^(z_il)) - z_ik)
    where:
    - N is the size of the data set,
    - K is the number of classes,
    - t_{ik} is the target value for data point i and class k,
    - z_{il} is the model output before softmax for data point i and class l,
    - z is the model output before softmax (matrix z).

    Parameters:
    inputs (np.ndarray): The input data.
    targets (np.ndarray): The target data.
    z (Union[np.ndarray, None]): The model output before softmax. If None, it will be computed.
    """

    # z - (n,k) matrix

    onehots = onehotencoding(targets, self.k)
    exp_sums = np.apply_along_axis(np.sum, axis=1, arr=np.apply_along_axis(np.exp, axis=1, arr=z))

    res: float = 0
    for i in range(len(inputs)):
        for k in range(self.k):
            t_ik = onehots[i][k]
            if t_ik == 1:
                res += math.log(exp_sums[i]) - z[i][k]

    return 1.0 * res
