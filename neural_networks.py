import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # Learning rate
        self.activation_fn = activation  # Activation function
        
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_dim, hidden_dim) * 0.1
        self.bias_hidden = np.zeros((1, hidden_dim))
        self.weights_hidden_output = np.random.randn(hidden_dim, output_dim) * 0.1
        self.bias_output = np.zeros((1, output_dim))

    def get_activation(self, name):
        activations = {
            'tanh': np.tanh,
            'relu': lambda x: np.maximum(0, x),
            'sigmoid': lambda x: 1 / (1 + np.exp(-x))
        }
        return activations[name]

    def get_activation_derivative(self, name):
        activation_derivatives = {
            'tanh': lambda x: 1 - np.tanh(x) ** 2,
            'relu': lambda x: (x > 0).astype(float),
            'sigmoid': lambda x: self.get_activation('sigmoid')(x) * (1 - self.get_activation('sigmoid')(x))
        }
        return activation_derivatives[name]

    def forward(self, X):
        self.hidden_linear = X @ self.weights_input_hidden + self.bias_hidden
        activation_func = self.get_activation(self.activation_fn)
        self.hidden_activation = activation_func(self.hidden_linear)
        self.output_linear = self.hidden_activation @ self.weights_hidden_output + self.bias_output
        sigmoid_func = self.get_activation('sigmoid')
        self.output_activation = sigmoid_func(self.output_linear)
        return self.output_activation

    def backward(self, X, y):
        m = X.shape[0]
        gradient_output_linear = self.output_activation - y
        gradient_weights_hidden_output = self.hidden_activation.T @ gradient_output_linear / m
        gradient_bias_output = np.sum(gradient_output_linear, axis=0, keepdims=True) / m
        
        gradient_hidden_activation = gradient_output_linear @ self.weights_hidden_output.T
        activation_derivative_func = self.get_activation_derivative(self.activation_fn)
        gradient_hidden_linear = gradient_hidden_activation * activation_derivative_func(self.hidden_linear)
        gradient_weights_input_hidden = X.T @ gradient_hidden_linear / m
        gradient_bias_hidden = np.sum(gradient_hidden_linear, axis=0, keepdims=True) / m

        self.weights_hidden_output -= self.lr * gradient_weights_hidden_output
        self.bias_output -= self.lr * gradient_bias_output
        self.weights_input_hidden -= self.lr * gradient_weights_input_hidden
        self.bias_hidden -= self.lr * gradient_bias_hidden

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform training steps
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)
        
    hidden_features = mlp.hidden_activation
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], c=y.ravel(), cmap='bwr', alpha=0.7)

    xx_grid, yy_grid = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 100), np.linspace(X[:, 1].min(), X[:, 1].max(), 100))
    input_grid = np.c_[xx_grid.ravel(), yy_grid.ravel()]

    ax_input.contourf(xx_grid, yy_grid, mlp.forward(input_grid).reshape(xx_grid.shape), levels=50, cmap="bwr", alpha=0.7)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="bwr", edgecolor="k")
    ax_input.set_title("Input Space Decision Boundary")

    for i, magnitude in enumerate(np.sqrt(np.sum(mlp.weights_input_hidden ** 2, axis=0))):
        circle = Circle((mlp.weights_input_hidden[0, i], mlp.weights_input_hidden[1, i]), magnitude, alpha=0.5)
        ax_gradient.add_patch(circle)
    ax_gradient.set_xlim(-1, 1)
    ax_gradient.set_ylim(-1, 1)
    ax_gradient.set_title("Gradients")

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)