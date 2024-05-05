"""Parametric UMAP loss function."""

from __future__ import annotations

import warnings

import torch

# isort: off
# Ignore warnings from Numba deprecation:
# https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-object-mode-fall-back-behaviour-when-using-jit
# Numba is required by UMAP.
from numba.core.errors import NumbaDeprecationWarning
from torch.nn.functional import mse_loss

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
from umap.umap_ import find_ab_params  # noqa : E402

# isort: on


def convert_distance_to_probability(distances, a=1.0, b=1.0):
    """Convert distances to probability.

    Computes equation (2.6) of Sainburg McInnes Gentner 2021,
    :math:`q_{ij} = (1 + a \abs{z_i - z_j}^{2b} )^{-1}`.

    The function uses torch.log1p to avoid floating point error:
    ``-torch.log1p(a * distances ** (2 * b))``.
    See https://en.wikipedia.org/wiki/Natural_logarithm#lnp1
    """
    # next line, equivalent to 1.0 / (1.0 + a * distances ** (2 * b))
    # but avoids floating point error
    return -torch.log1p(a * distances ** (2 * b))


def compute_cross_entropy(
    probabilities_graph,
    probabilities_distance,
    EPS=1e-4,
    repulsion_strength=1.0,
):
    """Computes cross entropy as used for UMAP cost function"""
    # cross entropy
    attraction_term = -probabilities_graph * torch.nn.functional.logsigmoid(
        probabilities_distance
    )
    repulsion_term = (
        -(1.0 - probabilities_graph)
        * (
            torch.nn.functional.logsigmoid(probabilities_distance)
            - probabilities_distance
        )
        * repulsion_strength
    )

    # balance the expected losses between attraction and repulsion
    CE = attraction_term + repulsion_term
    return attraction_term, repulsion_term, CE


def umap_loss(
    embedding_to: torch.Tensor,
    embedding_from: torch.Tensor,
    a,
    b,
    negative_sample_rate: int = 5,
):
    """UMAP loss function

    Converts distances to probabilities,
    and then computes cross entropy.
    """
    # get negative samples by randomly shuffling the batch
    embedding_neg_to = embedding_to.repeat(negative_sample_rate, 1)
    repeat_neg = embedding_from.repeat(negative_sample_rate, 1)
    embedding_neg_from = repeat_neg[torch.randperm(repeat_neg.shape[0])]
    distance_embedding = torch.cat(
        (
            (embedding_to - embedding_from).norm(dim=1),
            (embedding_neg_to - embedding_neg_from).norm(dim=1),
            # ``to`` method in next line to avoid error `Expected all tensors to be on the same device`
        ),
        dim=0,
    ).to(embedding_to.device)

    # convert probabilities to distances
    probabilities_distance = convert_distance_to_probability(
        distance_embedding, a, b
    )
    # set true probabilities based on negative sampling
    batch_size = embedding_to.shape[0]
    probabilities_graph = torch.cat(
        (
            torch.ones(batch_size),
            torch.zeros(batch_size * negative_sample_rate),
        ),
        dim=0,
        # ``to`` method in next line to avoid error `Expected all tensors to be on the same device`
    ).to(embedding_to.device)

    # compute cross entropy
    (attraction_loss, repellant_loss, ce_loss) = compute_cross_entropy(
        probabilities_graph,
        probabilities_distance,
    )
    loss = torch.mean(ce_loss)
    return loss


class UmapLoss(torch.nn.Module):
    """"""

    def __init__(
        self,
        spread: float = 1.0,
        min_dist: float = 0.1,
        negative_sample_rate: int = 5,
        beta: float = 1.0,
    ):
        super().__init__()
        self.min_dist = min_dist
        self.a, self.b = find_ab_params(spread, min_dist)
        self.negative_sample_rate = negative_sample_rate
        self.beta = beta

    def forward(
        self,
        embedding_to: torch.Tensor,
        embedding_from: torch.Tensor,
        reconstruction: torch.Tensor | None = None,
        before_encoding: torch.Tensor | None = None,
    ):
        loss_umap = umap_loss(
            embedding_to,
            embedding_from,
            self.a,
            self.b,
            self.negative_sample_rate,
        )
        if reconstruction is not None:
            loss_reconstruction = mse_loss(reconstruction, before_encoding)
            loss = loss_umap + self.beta * loss_reconstruction
        else:
            loss_reconstruction = None
            loss = loss_umap
        return loss_umap, loss_reconstruction, loss
