"""Pick-To-Learn (P2L), in Jax

The P2L method iteratively:
1. Trains a model on the current support set
2. Finds the "least appropriate" example from non-support set
3. Moves that example to the support set
4. Repeats until all remaining examples are "appropriate enough"

For more details, see the paper:
"The Pick-to-Learn Algorithm: Empowering Compression for Tight Generalization Bounds
and Improved Post-training Performance" -- Paccagnan, Campi, Garatti (2023)
https://openreview.net/forum?id=40L3viVWQN
"""

# TODO:
# - Improve re-compilation performance by better handling static shapes
#   Currently, the model is recompiled every time we change the support set size
#   Possible option: np.pad the data to a fixed size, and use an additional index argument
#   to determine where to slice. When needed, double the size of the array/buffer
# - Add logging for tracking loss and accuracy during training


import math
from functools import partial
from typing import Tuple, List, Dict, Any

from tqdm import trange
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax


@jax.tree_util.register_static
class P2LConfig:
    """Base configuration class for the Pick-to-Learn (P2L) method

    Classes inheriting from this should implement all NotImplemented methods

    Args:
        pretrain_fraction (float): Fraction of the dataset to use for pretraining
        max_iterations (int): Maximum number of iterations for P2L
        train_epochs (int): Number of epochs to train the model during each P2L training step
        batch_size (int): Batch size for training the model during each P2L training epoch
        confidence_param (float): P2L confidence parameter, used to compute the generalization bound
    """

    def __init__(
        self,
        pretrain_fraction: float,
        max_iterations: int,
        train_epochs: int,
        batch_size: int,
        confidence_param: float,
    ):
        assert isinstance(pretrain_fraction, float) and 0 <= pretrain_fraction <= 1
        assert isinstance(max_iterations, int) and max_iterations >= 1
        assert isinstance(train_epochs, int) and train_epochs >= 1
        assert isinstance(batch_size, int) and batch_size >= 1
        assert isinstance(confidence_param, float) and 0 <= confidence_param <= 1
        self.pretrain_fraction = pretrain_fraction
        self.max_iterations = max_iterations
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.confidence_param = confidence_param

    def init_model(self, key: jax.Array) -> nnx.Module:
        """Constructs an NNX model

        Args:
            key (jax.Array): JAX random key for model initialization

        Returns:
            model (nnx.Module): The model
        """
        raise NotImplementedError("Please implement the init_model method.")

    def init_optimizer(self) -> optax.GradientTransformation:
        """Constructs an Optax optimizer

        Returns:
            optimizer (optax.GradientTransformation): The optimizer
        """
        raise NotImplementedError("Please implement the init_optimizer method.")

    def init_data(self, key: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Loads the dataset

        NOTE: Currently, we assume that we are working with a dataset that we can
        fully load into memory

        Args:
            key (jax.Array): JAX random key

        Returns:
            (data, targets) (Tuple[jax.Array, jax.Array]): Tuple of
            - data: Training data, shape (n_samples, ...)
            - targets: Training data labels, shape (n_samples, ...)
        """
        raise NotImplementedError("Please implement the init_data method.")

    def loss_function(self, model_output: jax.Array, target: jax.Array) -> float:
        """Loss function.

        Args:
            model_output (jax.Array): Model outputs
            target (jax.Array): Target labels

        Returns:
            loss (float): Scalar loss value
        """
        raise NotImplementedError("Please implement the loss_function method.")

    def accuracy(self, model_output: jax.Array, target: jax.Array) -> float:
        """Accuracy function for the model.

        Args:
            model_output (jax.Array): Model outputs
            target (jax.Array): Target labels

        Returns:
            accuracy (float): Scalar accuracy value
        """
        raise NotImplementedError("Please implement the accuracy method.")

    @partial(jax.jit, static_argnames=["self"])
    def eval_p2l_convergence(
        self, model_output: jax.Array, target: jax.Array
    ) -> Tuple[int, bool]:
        """Evaluate convergence of pick-to-learn, and (if not converged),
        the worst datapoint to be added to the support set

        Args:
            model_output (jax.Array): Model outputs
            target (jax.Array): Target labels

        Returns:
            (worst_index, converged) (Tuple[int, bool]): Tuple of
            - Index of the worst example in the non-support set
            - Whether the P2L algorithm has converged
        """
        raise NotImplementedError("Please implement the eval_p2l_convergence method.")

    # NOTE: Keep model_state as the first argument since the gradient will be computed with respect to it
    def loss_and_aux(
        self,
        model_state: nnx.State,
        graphdef: nnx.GraphDef,
        data: jax.Array,
        target: jax.Array,
        deterministic: bool,
        key: jax.Array,
    ) -> Tuple[float, Tuple[float, jax.Array]]:
        """Loss function and auxiliary outputs for the model.

        Args:
            model_state (nnx.State): Current model parameter state
            graphdef (nnx.GraphDef): Model graph definition (static)
            data (jax.Array): Input data
            target (jax.Array): Target labels
            deterministic (bool): Whether to run in deterministic mode
                (e.g., disabling dropout/stochasticity for evaluation)
            key (jax.Array): Jax random key

        Returns:
            (loss, aux) (Tuple[float, Tuple[float, jax.Array]]): Tuple of
            - loss: Scalar loss value
            - Auxiliary outputs, a tuple of (accuracy, model_output) where
                - accuracy: Model accuracy on the batch
                - model_output: Model outputs for further use downstream in P2L
        """
        model_output = self.forward(graphdef, model_state, data, deterministic, key)
        loss = self.loss_function(model_output, target)
        accuracy = self.accuracy(model_output, target)
        return loss, (accuracy, model_output)

    def forward(
        self,
        graphdef: nnx.GraphDef,
        model_state: nnx.State,
        data: jax.Array,
        deterministic: bool,
        key: jax.Array,
    ) -> jax.Array:
        """Forward pass through the model.

        Args:
            graphdef (nnx.GraphDef): Model graph definition (static)
            model_state (nnx.State): Current model parameter state
            data (jax.Array): Input data
            deterministic (bool): Whether to run in deterministic mode
                (e.g., disabling dropout/stochasticity for evaluation)
            key (jax.Array): Jax random key

        Returns:
            model_outputs (jax.Array): Result of forward pass
        """
        model = nnx.merge(graphdef, model_state)
        return model(data, deterministic=deterministic, key=key)

    @partial(jax.jit, static_argnames=["self", "graphdef", "optimizer"])
    def train_step(
        self,
        graphdef: nnx.GraphDef,
        model_state: nnx.State,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        data: jax.Array,
        target: jax.Array,
        key: jax.Array,
    ) -> Tuple[float, float, nnx.State, optax.OptState]:
        """Perform a single training step on a batch of data.

        Args:
            graphdef (nnx.GraphDef): Model graph definition (static)
            model_state (nnx.State): Current model parameter state
            optimizer (optax.GradientTransformation): Optax optimizer (static)
            opt_state (optax.OptState): Current optimizer state
            data (jax.Array): Input data
            target (jax.Array): Target labels
            key (jax.Array): Jax random key

        Returns:
            (loss, accuracy, model_state, opt_state) (Tuple[float, float, nnx.State, optax.OptState]): Tuple of
            - loss: Scalar loss value
            - accuracy: Model accuracy on the batch
            - model_state: Updated model parameter state after training
            - opt_state: Updated optimizer state after training
        """
        deterministic = False
        (loss, aux), grads = jax.value_and_grad(
            self.loss_and_aux, argnums=0, has_aux=True
        )(model_state, graphdef, data, target, deterministic, key)
        accuracy, _model_output = aux  # Unpack auxiliary outputs
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_model_state = optax.apply_updates(model_state, updates)
        return loss, accuracy, new_model_state, new_opt_state

    @partial(jax.jit, static_argnames=["self", "graphdef"])
    def eval_step(
        self,
        graphdef: nnx.GraphDef,
        model_state: nnx.State,
        data: jax.Array,
        target: jax.Array,
    ) -> Tuple[float, float, jax.Array]:
        """Evaluate the model on a batch of data.

        Args:
            graphdef (nnx.GraphDef): Model graph definition (static)
            model_state (nnx.State): Current model parameter state
            data (jax.Array): Input data
            target (jax.Array): Target labels

        Returns:
            (loss, accuracy, model_output) (Tuple[float, float, jax.Array]): Tuple of
            - loss: Scalar loss value
            - accuracy: Model accuracy on the batch
            - model_output: Model outputs for further use downstream in P2L
        """
        deterministic = True  # Evaluation is always deterministic
        # No need for RNGs during evaluation. Pass in a dummy value
        # TODO figure out a better way to handle this?
        key = jax.random.key(0)
        loss, aux = self.loss_and_aux(
            model_state, graphdef, data, target, deterministic, key
        )
        accuracy, model_output = aux  # Unpack auxiliary outputs
        return loss, accuracy, model_output

    def train_on_support_set(
        self,
        graphdef: nnx.GraphDef,
        model_state: nnx.State,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        support_data: jax.Array,
        support_targets: jax.Array,
        key: jax.Array,
    ) -> Tuple[nnx.State, optax.OptState]:
        """Train the model on the current support set

        Args:
            graphdef (nnx.GraphDef): Model graph definition (static)
            model_state (nnx.State): Current model parameter state
            optimizer (optax.GradientTransformation): Optax optimizer (static)
            opt_state (optax.OptState): Current optimizer state
            support_data (jax.Array): Support set data
            support_targets (jax.Array): Support set target labels
            key (jax.Array): Jax random key

        Returns:
            (model_state, opt_state) (Tuple[nnx.State, optax.OptState]): Tuple of
            - model_state: Updated model parameter state after training
            - opt_state: Updated optimizer state after training
        """
        num_support = support_data.shape[0]

        # Train for specified epochs
        for _epoch in trange(
            self.train_epochs, desc="Training on support set", leave=False
        ):
            key, key_epoch = jax.random.split(key)
            # Shuffle data for this epoch
            perm = jax.random.permutation(key_epoch, num_support)
            data_shuffled = support_data[perm]
            targets_shuffled = support_targets[perm]

            # Train in batches
            num_batches = math.ceil(num_support / self.batch_size)
            for batch_idx in range(num_batches):
                key, key_batch = jax.random.split(key)

                start = batch_idx * self.batch_size
                end = min((batch_idx + 1) * self.batch_size, data_shuffled.shape[0])

                batch_data = data_shuffled[start:end]
                batch_targets = targets_shuffled[start:end]

                # Perform training step
                _loss, _accuracy, model_state, opt_state = self.train_step(
                    graphdef,
                    model_state,
                    optimizer,
                    opt_state,
                    batch_data,
                    batch_targets,
                    key_batch,
                )

        return model_state, opt_state

    @partial(jax.jit, static_argnames=["self", "graphdef"])
    def evaluate_on_nonsupport_set(
        self,
        graphdef: nnx.GraphDef,
        model_state: nnx.State,
        nonsupport_data: jax.Array,
        nonsupport_targets: jax.Array,
    ) -> Tuple[float, float, int, bool]:
        """Evaluate current model performance on the non-support set.

        Args:
            graphdef (nnx.GraphDef): Model graph definition (static)
            model_state (nnx.State): Current model parameter state
            nonsupport_data (jax.Array): Non-support set data
            nonsupport_targets (jax.Array): Non-support set target labels

        Returns:
            (loss, accuracy, worst_index, converged) (Tuple[float, float, int, bool]): Tuple of
            - loss: Scalar loss value on the non-support set
            - accuracy: Model accuracy on the non-support set
            - worst_index: Index of the worst example in the non-support set
            - converged: Whether the P2L algorithm has converged
        """
        loss, accuracy, model_output = self.eval_step(
            graphdef, model_state, nonsupport_data, nonsupport_targets
        )
        worst_index, converged = self.eval_p2l_convergence(
            model_output, nonsupport_targets
        )
        return loss, accuracy, worst_index, converged


def initialize_support_sets(
    n_total: int, pretrain_fraction: float, key: jax.Array
) -> Tuple[List[int], List[int]]:
    """Initialize support and non-support sets.

    Args:
        n_total (int): Total number of data points
        pretrain_fraction (float): Fraction of data to use for initial support set
        key (jax.Array): Jax random key

    Returns:
        (suport_indices, nonsupport_indices) (Tuple[List[int], List[int]]): Tuple of
        - support_indices: Initial examples for pretraining the model
        - nonsupport_indices: Remaining examples to be evaluated
    """
    perm = jax.random.permutation(key, n_total)
    n_pretrain = int(pretrain_fraction * n_total)
    support_indices = perm[:n_pretrain].tolist()
    nonsupport_indices = perm[n_pretrain:].tolist()
    return support_indices, nonsupport_indices


def generalization_bound(k: int, N: int, beta: float) -> float:
    """Computes the P2L generalization bound

    Args:
        k (int): Number of datapoints added to the support set
        N (int): Total number of examples in the dataset that were not used for pretraining
        beta (float): Confidence parameter: with confidence 1 - beta, the risk is bounded by epsilon

    Returns:
        float: Generalization bound for the P2L algorithm, epsilon
    """
    log_m_choose_k = [
        math.lgamma(m + 1) - math.lgamma(m - k + 1) - math.lgamma(k + 1)
        for m in range(k, N)
    ]
    log_N_choose_k = math.lgamma(N + 1) - math.lgamma(N - k + 1) - math.lgamma(k + 1)
    coeffs = np.array(log_m_choose_k) - np.array([log_N_choose_k] * (N - 1 - k + 1))
    m_vec = np.array([m for m in range(k, N)])

    t1 = 0
    t2 = 1
    while t2 - t1 > 1e-10:
        t = (t1 + t2) / 2
        val = 1 - (beta / (N) * np.sum(np.exp(coeffs - (N - m_vec) * np.log(t))))
        if val > 0:
            t2 = t
        else:
            t1 = t
    eps = 1 - t1
    return eps


def pick_to_learn(config: P2LConfig, key: jax.Array) -> Dict[str, Any]:
    """Pick-to-Learn (P2L) Method

    The P2L method iteratively:
    1. Trains a model on the current support set
    2. Finds the "least appropriate" example from non-support set
    3. Moves that example to the support set
    4. Repeats until all remaining examples are "appropriate enough"

    Args:
        config (P2LConfig): P2L configuration for your choice of dataset and model / hypothesis class
        key (jax.Array): Jax random key

    Returns:
        results (Dict[str, Any]): A dictionary containing
        - final_model (nnx.Module): Final trained model after P2L
        - support_indices (List[int]): List of final support set indices
        - nonsupport_indices (List[int]): List of final non-support set indices
        - generalization_bound (float): Computed P2L generalization bound
        - num_iterations (int): Number of P2L iterations performed
        - converged (bool): Whether P2L converged
        - losses (List[float]): List of losses across P2L iterations
        - accuracies (List[float]): List of accuracies across P2L iterations
    """
    # Build model, optimizer, data
    key, model_key, data_key, sets_key = jax.random.split(key, 4)
    model = config.init_model(model_key)
    graphdef, model_state = nnx.split(model)
    opt = config.init_optimizer()
    opt_state = opt.init(model_state)
    data, targets = config.init_data(data_key)

    n_total = data.shape[0]

    # Initialize support and non-support sets
    support_indices, nonsupport_indices = initialize_support_sets(
        n_total, config.pretrain_fraction, sets_key
    )
    initial_support_set_size = len(support_indices)
    initial_nonsupport_set_size = len(nonsupport_indices)

    print(f"Starting P2L with {len(support_indices)} initial support examples")

    iteration = 0
    converged = False

    # Logging metrics
    losses = []
    accuracies = []

    # Main P2L loop
    while iteration < config.max_iterations:
        print(f"\n--- P2L Iteration {iteration + 1} ---")
        print(f"Support set size: {len(support_indices)}")
        print(f"Non-support set size: {len(nonsupport_indices)}")

        key, key_train = jax.random.split(key)

        # Step 1: Train model on current support set
        if support_indices:  # Check edge case for empty support set
            model_state, opt_state = config.train_on_support_set(
                graphdef,
                model_state,
                opt,
                opt_state,
                data[np.array(support_indices)],
                targets[np.array(support_indices)],
                key_train,
            )

        # Step 2: Evaluate on non-support set and check convergence
        loss, accuracy, worst_nonsupport_index, converged = (
            config.evaluate_on_nonsupport_set(
                graphdef,
                model_state,
                data[np.array(nonsupport_indices)],
                targets[np.array(nonsupport_indices)],
            )
        )

        # Log metrics
        losses.append(loss)
        accuracies.append(accuracy)
        print(f"Loss: {loss}")
        print(f"Accuracy: {accuracy}")

        if converged:
            print("P2L Converged!")
            break

        # Step 3: If not converged, move least appropriate example to support set
        support_indices.append(nonsupport_indices.pop(worst_nonsupport_index))

        iteration += 1

    # Check final convergence
    if not converged:
        print(f"Warning: P2L did not converge after {config.max_iterations} iterations")

    # Calculate generalization bound
    bound = generalization_bound(
        len(support_indices) - initial_support_set_size,
        initial_nonsupport_set_size,
        config.confidence_param,
    )

    return {
        "final_model": nnx.merge(graphdef, model_state),
        "support_indices": support_indices,
        "nonsupport_indices": nonsupport_indices,
        "generalization_bound": bound,
        "num_iterations": iteration,
        "converged": converged,
        "losses": losses,
        "accuracies": accuracies,
    }
