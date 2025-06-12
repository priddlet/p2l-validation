"""Binary MNIST P2L Example

This is based directly on the original P2L MNIST example, but adapted to use JAX.
See the original P2L code (torch) for more details: the code can be found in the
suppolementary materials of the paper: https://openreview.net/forum?id=40L3viVWQN

(this is not intended to be a great MNIST classifier, but rather a demo of P2L)
"""

from typing import Union

from torchvision import datasets, transforms
import jax
import jax.numpy as jnp
from flax import nnx
import optax

from p2l import P2LConfig, pick_to_learn


@jax.tree_util.register_static
class BinaryMNISTP2LConfig(P2LConfig):
    """P2L Configuration for the Binary MNIST example

    Args:
        n_datapoints (int): Number of data points to use out of the full MNIST dataset
        dataset_slice_index (int): For slices of the dataset (n_datapoints < full dataset size),
            this is the starting slice index of a random permutation of the dataset
        dropout_prob (float): Dropout probability for the network
        learning_rate (float): Learning rate for the optimizer
        momentum (float): Momentum for the optimizer
        convergence_param (float): P2L convergence parameter for this MNIST example. How to define
            convergence is up to the designer, but in this case, the original P2L code defined it
            as the max loss on any datapoint being less than this value.
        pretrain_fraction (float): Fraction of the dataset to use for pretraining
        max_iterations (int): Maximum number of iterations for P2L
        train_epochs (int): Number of epochs to train the model during each P2L training step
        batch_size (int): Batch size for training the model during each P2L training epoch
        confidence_param (float): P2L confidence parameter, used to compute the generalization bound
    """

    def __init__(
        self,
        n_datapoints: int,
        dataset_slice_index: int,
        dropout_prob: float,
        learning_rate: float,
        momentum: float,
        convergence_param: float,
        pretrain_fraction: float,
        max_iterations: int,
        train_epochs: int,
        batch_size: int,
        confidence_param: float,
    ):
        # MNIST-Specific params
        assert isinstance(n_datapoints, int) and n_datapoints > 0
        assert isinstance(dataset_slice_index, int) and dataset_slice_index >= 0
        assert isinstance(dropout_prob, float) and 0 <= dropout_prob <= 1
        assert isinstance(learning_rate, float) and learning_rate >= 0
        assert isinstance(momentum, float) and momentum >= 0
        assert isinstance(convergence_param, float)
        self.n_datapoints = n_datapoints
        self.dataset_slice_index = dataset_slice_index
        self.dropout_prob = dropout_prob
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.convergence_param = convergence_param

        # P2L params
        super().__init__(
            pretrain_fraction,
            max_iterations,
            train_epochs,
            batch_size,
            confidence_param,
        )

    def init_model(self, key):
        return BinaryMNISTExampleModel(self.dropout_prob, key=key)

    def init_optimizer(self):
        return optax.sgd(self.learning_rate, self.momentum)

    def init_data(self, key):
        # Loading data the same way as in the original P2L code
        # Note: they didn't actually use the test set, so I excluded it here
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train = datasets.MNIST(
            "mnist-data/", train=True, download=True, transform=transform
        )
        train.targets = (train.targets > 4).float()  # Binarize
        data = (train.data.float() / 255 - 0.1307) / 0.3081  # rescale
        targets = train.targets.long()

        # Select a subset of the dataset
        assert self.dataset_slice_index + self.n_datapoints <= data.shape[0]
        permutation = jax.random.permutation(key, data.shape[0])
        idxs = permutation[
            self.dataset_slice_index : self.dataset_slice_index + self.n_datapoints
        ]
        return data.numpy()[idxs], targets.numpy()[idxs]

    def loss_function(self, model_output, target):
        # TODO: consider using one of the built in optax losses?
        return nll_loss(model_output, target, reduction="mean")

    def accuracy(self, model_output: jax.Array, target: jax.Array) -> float:
        return jnp.mean(jnp.argmax(model_output, axis=-1) == target)

    def eval_p2l_convergence(self, model_output, target):
        elementwise_losses = nll_loss(model_output, target, reduction="none")
        max_loss_index = jnp.argmax(elementwise_losses)
        converged = elementwise_losses[max_loss_index] <= self.convergence_param
        return max_loss_index, converged


# TODO: Comfirm that shapes are correct for the inputs
def nll_loss(
    log_probs: jax.Array, targets: jax.Array, reduction: str
) -> Union[float, jax.Array]:
    """Negative log likelihood loss function.

    Args:
        log_probs (jax.Array): Log probabilities, shape (batch_size, num_classes)
        targets (jax.Array): Class labels, shape (batch_size,)
        reduction (str): 'mean', 'sum', or 'none'

    Returns:
        loss (float or jax.Array):
        - If reduction is 'mean', returns the mean loss across the batch (float).
        - If reduction is 'sum', returns the sum of losses across the batch (float).
        - If reduction is 'none', returns an array of per-sample losses (jax.Array).
    """
    one_hot_targets = jax.nn.one_hot(targets, num_classes=log_probs.shape[-1])
    losses = -jnp.sum(log_probs * one_hot_targets, axis=-1)

    if reduction == "mean":
        return losses.mean()
    elif reduction == "sum":
        return losses.sum()
    elif reduction == "none":
        return losses
    else:
        raise ValueError(
            f"Invalid reduction: {reduction}. Must be 'mean', 'sum', or 'none'"
        )


class BinaryMNISTExampleModel(nnx.Module):
    """A 4-layer fully connected neural network, used for the Binary MNIST example.

    Args:
        dropout_prob (float): The dropout probability.
        key (jax.Array): Jax random key for parameter initialization.
    """

    def __init__(self, dropout_prob: float, key: jax.Array):
        param_keys = jax.random.split(key, 4)
        self.l1 = self._build_linear(28 * 28, 600, key=param_keys[0])
        self.l2 = self._build_linear(600, 600, key=param_keys[1])
        self.l3 = self._build_linear(600, 600, key=param_keys[2])
        self.l4 = self._build_linear(600, 2, key=param_keys[3])
        self.dropout = nnx.Dropout(rate=dropout_prob)

    @staticmethod
    def _build_linear(
        in_features: int, out_features: int, key: jax.Array
    ) -> nnx.Linear:
        """Helper function:
        Creates a linear layer with truncated normal initialization on the weights

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            key (jax.Array): Jax random key for parameter initialization.

        Returns:
            layer (nnx.Linear): A linear layer initialized with truncated normal weights.
        """
        sigma_weights = 1.0 / jnp.sqrt(in_features)
        # Note: bias initialization defaults to zeros in nnx.Linear
        return nnx.Linear(
            in_features=in_features,
            out_features=out_features,
            kernel_init=nnx.initializers.truncated_normal(
                stddev=sigma_weights, lower=-2 * sigma_weights, upper=2 * sigma_weights
            ),
            rngs=nnx.Rngs(params=key),
        )

    def __call__(self, x: jax.Array, deterministic: bool, key: jax.Array) -> jax.Array:
        """Forward pass through the network.

        Args:
            x (jax.Array): Input data
            deterministic (bool): If True, dropout is not applied (for evaluation).
            key (jax.Array): Jax random key for dropout.

        Returns:
            model_output (jax.Array): Log probabilities of the output classes.
        """
        keys = jax.random.split(key, 3)

        x = x.reshape((x.shape[0], -1))  # Flatten input

        x = self.l1(x)
        x = self.dropout(x, deterministic=deterministic, rngs=nnx.Rngs(dropout=keys[0]))
        x = jax.nn.relu(x)

        x = self.l2(x)
        x = self.dropout(x, deterministic=deterministic, rngs=nnx.Rngs(dropout=keys[1]))
        x = jax.nn.relu(x)

        x = self.l3(x)
        x = self.dropout(x, deterministic=deterministic, rngs=nnx.Rngs(dropout=keys[2]))
        x = jax.nn.relu(x)

        x = self.l4(x)
        return jax.nn.log_softmax(x, axis=-1)


def main():
    """Main function to run the Binary MNIST P2L example."""

    # Initialize random seed for reproducibility
    key = jax.random.key(0)

    # Construct P2L Config
    config = BinaryMNISTP2LConfig(
        n_datapoints=1000,
        dataset_slice_index=0,
        dropout_prob=0.2,
        learning_rate=0.01,
        momentum=0.95,
        convergence_param=0.69314718,  # -ln(0.5)
        pretrain_fraction=0.5,
        max_iterations=1000,
        train_epochs=200,
        batch_size=1000,
        confidence_param=0.035,
    )

    # Run P2L
    result = pick_to_learn(config, key)

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Converged: {result['converged']}")
    print(f"Iterations: {result['num_iterations']}")
    print(f"Final support set size: {len(result['support_indices'])}")
    print(f"Generalization bound: {result['generalization_bound']:.5f}")


if __name__ == "__main__":
    main()
