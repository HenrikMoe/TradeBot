import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as mpl
from sklearn.preprocessing import scale

# Custom MLP implementation functions
def _CreateMLP(_X, _W, _B, _AF):
    n = len(_W)
    for i in range(n - 1):
        if _X.shape[1] != _W[i].shape[0]:
            raise ValueError("Matrix dimensions are incompatible.")
        _X = _AF(tf.matmul(_X, _W[i]) + _B[i])

    if _X.shape[1] != _W[n - 1].shape[0]:
        raise ValueError("Matrix dimensions are incompatible.")
    return tf.matmul(_X, _W[n - 1]) + _B[n - 1]

def _CreateL2Reg(_W, _B):
    n = len(_W)
    regularizers = tf.nn.l2_loss(_W[0]) + tf.nn.l2_loss(_B[0])
    for i in range(1, n):
        regularizers += tf.nn.l2_loss(_W[i]) + tf.nn.l2_loss(_B[i])
    return regularizers

def _CreateVars(layers):
    weight = []
    bias = []
    n = len(layers)
    for i in range(n - 1):
        lyrstd = np.sqrt(1.0 / layers[i])
        curW = tf.Variable(tf.random.normal([layers[i], layers[i + 1]], stddev=lyrstd))
        weight.append(curW)
        curB = tf.Variable(tf.random.normal([layers[i + 1]], stddev=lyrstd))
        bias.append(curB)
    return (weight, bias)

class SeqMLP(tf.keras.Model):
    def __init__(self, input_size, hidden_size, output_size, actvFn='tanh', learnRate=0.001, maxItr=2000, tol=1e-2, verbose=False, reg=0.001):
        super(SeqMLP, self).__init__()
        self.tol = tol
        self.mItr = maxItr
        self.vrbse = verbose
        self.reg = reg  # Added the reg as an instance variable

        self.weight, self.bias = _CreateVars([input_size] + hidden_size + [output_size])
        self.optmzr = tf.optimizers.Adam(learning_rate=learnRate)

    def _create_model(self, X):
        n = len(self.weight)
        for i in range(n - 1):
            if X.shape[1] != self.weight[i].shape[0]:
                raise ValueError(f"Matrix dimensions are incompatible at layer {i + 1}: {X.shape} vs {self.weight[i].shape}")

            X = tf.nn.tanh(tf.matmul(X, self.weight[i]) + self.bias[i])

        if X.shape[1] != self.weight[n - 1].shape[0]:
            raise ValueError(f"Matrix dimensions are incompatible at output layer: {X.shape} vs {self.weight[n - 1].shape}")

        return tf.matmul(X, self.weight[n - 1]) + self.bias[n - 1]

    def _l2_loss(self, y_true, y_pred):
        return tf.reduce_sum(tf.nn.l2_loss(y_true - y_pred))

    def _l2_regularization(self):
        regularizers = tf.nn.l2_loss(self.weight[0]) + tf.nn.l2_loss(self.bias[0])
        for i in range(1, len(self.weight)):
            regularizers += tf.nn.l2_loss(self.weight[i]) + tf.nn.l2_loss(self.bias[i])
        return regularizers

    def _train_step(self, X, y):
        with tf.GradientTape() as tape:
            pred = self._create_model(X)
            loss = self._l2_loss(y, pred)
            if self.reg is not None:
                loss += self._l2_regularization() * self.reg

        trainable_variables = self.weight + self.bias  # Collect all trainable variables

        gradients = tape.gradient(loss, trainable_variables)

        # Check dimensions of gradients and variables at each layer
        for i, (grad, var) in enumerate(zip(gradients, trainable_variables)):
            if grad.shape != var.shape:
                raise ValueError(f"Gradient and variable shapes are incompatible at layer {i}:\n"
                                 f"Gradient shape: {grad.shape}\n"
                                 f"Variable shape: {var.shape}")

        # Recreate optimizer instance with a constant learning rate
        learning_rate = 0.001  # You can adjust this value if needed
        self.optmzr = tf.optimizers.Adam(learning_rate=learning_rate)

        # Apply gradients to all trainable variables
        self.optmzr.apply_gradients(zip(gradients, trainable_variables))

        return loss

    def fit(self, A, y):
        m, _ = A.shape
        A = tf.convert_to_tensor(A, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)

        for i in range(self.mItr):
            loss = self._train_step(A, y)

            err = np.sqrt(loss * 2.0 / m)
            if self.vrbse:
                print("Iter " + str(i + 1) + ": " + str(err))
            if err < self.tol:
                print("Converged at iteration " + str(i + 1))
                break

    def predict(self, A):
        A = tf.convert_to_tensor(A, dtype=tf.float32)
        return self._create_model(A).numpy()

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the CSV file name
csv_filename = 'Skanky Stakes - BTC.csv'

# Create the full path to the CSV file
csv_path = os.path.join(script_dir, csv_filename)

# Load the CSV file with only low and high prices (columns 1 and 4)
A = np.genfromtxt(csv_path, delimiter=",", skip_header=1, usecols=(0, 1), invalid_raise=False)

# Check if the data needs reshaping
if A.ndim == 1:
    A = A.reshape(-1, 1)

# Print raw low prices (col1) and high prices (col2) from the CSV
print("Raw Low Prices:", A[:, 0])
print("Raw High Prices:", A[:, 1])

# Map the low prices (col1) to low_prices
low_prices = A[:, 0]
# Map the high prices (col2) to high_prices
high_prices = A[:, 1]
# y is the dependent variable
y = A[:, 1].reshape(-1, 1)
# Handle NaN values with more sophisticated imputation
low_prices = np.nan_to_num(low_prices, nan=np.nanmean(low_prices))
high_prices = np.nan_to_num(high_prices, nan=np.nanmean(high_prices))
y = np.nan_to_num(y, nan=np.nanmean(y))
# Use the same scaling factors for both training and prediction
# Note: In a real-world scenario, you would save the scaling factors during training and use them for prediction.
# For simplicity, I'm assuming here that the range of the scaling factors is similar in both cases.
A_scaled = scale(A)

# Example layer sizes
layers = [1, 32, 32, 1]

# Create the custom SeqMLP instance
seq_mlp = SeqMLP(input_size=2, hidden_size=[32, 32], output_size=1, actvFn='tanh', learnRate=0.001, maxItr=2000, tol=1e-2, verbose=False, reg=0.001)

# Length of the hold-out period
nDays = 0
n = len(A)

# Learn the data
seq_mlp.fit(A_scaled[0:(n-nDays)], y[0:(n-nDays)])

# Begin prediction
yHat = seq_mlp.predict(A_scaled)

print("Predicted Prices:", yHat[:, 0])
print("Number of Predicted Prices:", len(yHat))

# Plot the original low prices with day numbers
mpl.plot(range(1, len(low_prices) + 1), low_prices, label='Original Low Prices', c='#ff5733')

# Plot the original high prices with day numbers
mpl.plot(range(1, len(high_prices) + 1), high_prices, label='Original High Prices', c='#33ff57')

# Plot the predicted prices with day numbers
mpl.plot(range(1, len(yHat) + 1), yHat[:, 0], label='Predicted Prices', c='#576cff')

mpl.xlabel('Days')
mpl.ylabel('Price')
mpl.title('Original and Predicted Prices Over Time')
mpl.legend()
mpl.show()
