import pathlib
import json
from typing import Callable, Literal, Optional
from tqdm.notebook import tqdm

import numpy as np
import jax
from jax import numpy as jnp, jit, scipy as jsp
import ot
import ott

from src import point_arrangement


def sphere_gw_distance(m: int, n: int) -> float:
    """Computes the closed form expression for the (4, 2)-Gromov Wasserstein
    distance between the Euclidean `m`- and `n`-spheres.

    Args:
        m (int): Dimensionality of the first sphere.
        n (int): Dimensionality of the second sphere.

    Returns:
        float: The (4, 2)-Gromov Wasserstein distance between the spheres.
    """
    if m > n:
        m, n = n, m
    return (
        (1 / jnp.sqrt(2))
        * jnp.power(
            (1 / (m + 1))
            + (1 / (n + 1))
            - (
                (2 / (m + 1))
                * (
                    (
                        (
                            jsp.special.gamma((m + 2) / 2)
                            * jsp.special.gamma((n + 1) / 2)
                        )
                        / (
                            jsp.special.gamma((m + 1) / 2)
                            * jsp.special.gamma((n + 2) / 2)
                        )
                    )
                    ** 2
                )
            ),
            0.25,
        )
    ).item()


cost_matrix: Callable[[jnp.ndarray], jnp.ndarray] = lambda pts: jnp.sum(
    (pts[:, None, :] - pts[None, :, :]) ** 2, axis=-1
)
"""Computes the Euclidean distance squared between each pair of rows
of the inputted matrix, pts. The outputted matrix satisfies M[i, j] = 
||pts[i] - pts[j]||^2.

Args:
    pts (jnp.ndarray): The matrix with data vectors as rows.

Returns:
    jnp.ndarray: the resulting cost matrix, M.
"""


def validate_weights(pts: jnp.ndarray, wghts: Optional[jnp.ndarray]) -> jnp.ndarray:
    """Checks that the given weights have the correct properties. Additionally,
    if the inputted weights are None, then it changes the weights to the
    uniform distribution.

    Args:
        pts (jnp.ndarray): An array of the shape [n_points, n_dimensions] with
        the coordinates of the points.
        wghts (Optional[jnp.ndarray]): An array of the shape [n_points] with
        the weights of the points. Its contents should sum to 1.

    Returns:
        jnp.ndarray: The validated weights (also of shape [n_points]).
    """
    wghts = jnp.ones(pts.shape[0]) / pts.shape[0] if wghts is None else wghts

    assert jnp.allclose(jnp.array(1.0), wghts.sum())
    assert pts.shape[0] == wghts.shape[0]

    return wghts


def get_empirical_gw_pot(
    pts_1: jnp.ndarray,
    pts_2: jnp.ndarray,
    epsilon: float = -1.0,
    wghts_1: Optional[jnp.ndarray] = None,
    wghts_2: Optional[jnp.ndarray] = None,
) -> float:
    """Computes the empirical (4, 2)-Gromov Wasserstein distance in Euclidean
    spaces using the Python Optimal Transport Package.

    Args:
        pts_1 (jnp.ndarray): The first set of points in Euclidean space.
        pts_2 (jnp.ndarray): The second set of points in Euclidean space.
        epsilon (float, optional): Regularization parameter for solver. Does
        unregularized OT if epsilon is less than zero. Defaults to -1.0.

    Returns:
        float: The empirical (4, 2)-Gromov Wasserstein distance.
    """
    wghts_1 = validate_weights(pts_1, wghts_1)
    wghts_2 = validate_weights(pts_2, wghts_2)

    n_1, n_2 = pts_1.shape[0], pts_2.shape[0]
    if epsilon < 0:
        T, log = ot.gromov.gromov_wasserstein(
            np.array(cost_matrix(pts_1)),
            np.array(cost_matrix(pts_2)),
            np.array(wghts_1),
            np.array(wghts_2),
            "square_loss",
            log=True,
        )
    else:
        T, log = ot.gromov.entropic_gromov_wasserstein(
            np.array(cost_matrix(pts_1)),
            np.array(cost_matrix(pts_2)),
            np.array(wghts_1),
            np.array(wghts_2),
            "square_loss",
            epsilon=epsilon,
            log=True,
        )
    return (1 / 2) * (log["gw_dist"] ** 0.25)


def get_empirical_gw_ott(
    pts_1: jnp.ndarray,
    pts_2: jnp.ndarray,
    epsilon: float = 1e-2,
    wghts_1: Optional[jnp.ndarray] = None,
    wghts_2: Optional[jnp.ndarray] = None,
) -> float:
    """Computes the empirical (4, 2)-Gromov Wasserstein distance in Euclidean
    spaces using the Optimal Transport Tools Package.

    Args:
        pts_1 (jnp.ndarray): The first set of points in Euclidean space.
        pts_2 (jnp.ndarray): The second set of points in Euclidean space.
        epsilon (float, optional): Regularization parameter for solver.
        Defaults to 1e-2.

    Returns:
        float: The empirical (4, 2)-Gromov Wasserstein distance.
    """
    wghts_1 = validate_weights(pts_1, wghts_1)
    wghts_2 = validate_weights(pts_2, wghts_2)

    geom_1 = ott.geometry.pointcloud.PointCloud(
        pts_1, pts_1, cost_fn=ott.geometry.costs.SqEuclidean()
    )
    geom_2 = ott.geometry.pointcloud.PointCloud(
        pts_2, pts_2, cost_fn=ott.geometry.costs.SqEuclidean()
    )
    prob = ott.problems.quadratic.quadratic_problem.QuadraticProblem(
        geom_1, geom_2, a=wghts_1, b=wghts_2
    )
    solver = jit(
        ott.solvers.quadratic.gromov_wasserstein.GromovWasserstein(
            warm_start=True, relative_epsilon=True, epsilon=epsilon
        )
    )
    soln: ott.solvers.quadratic.gromov_wasserstein.GWOutput = solver(prob)

    return (1 / 2) * (soln.primal_cost.item() ** 0.25)


def arranged_trial(d_1, d_2, n_1, n_2, seed=0, epsilon=1e-2, arranging_iters=1000):
    """Samples `n_1` and `n_2` points from the Euclidean sphere in `d_1` and
    `d_2` dimensions and computes their GW distance using the POT and OTT
    packages. This trial uses the "arranging" sampling method, where each point
    is modeled as a positive charge and the physics is modeled to push them away
    from each other.

    Args:
        d_1 (int): The ambient dimension of the first sphere (the intrinsic
        is one less)
        d_2 (int): The ambient dimension of the second sphere (the intrinsic
        is one less)
        n_1 (int): The number of points to sample from the first sphere.
        n_2 (int): The number of points to sample from the second sphere.
        seed (int, optional): Seed for randomly sample points. Defaults to 0.
        epsilon (_type_, optional): Regularization for the optimal transport
        solvers. Defaults to 1e-2.
        arranging_iters (int, optional): Number of time steps to use in the
        arrangement physics simulation. Defaults to 1000.

    Returns:
        dict[str, float]: Dictionary containing the values of the two
        solvers as well as the true distance.
    """
    rng = jax.random.PRNGKey(seed)
    pts_1, _, _ = point_arrangement.get_arranged_sphere_pts(
        d_1, n_1, rng=rng, n_iters=arranging_iters
    )
    pts_2, _, _ = point_arrangement.get_arranged_sphere_pts(
        d_2, n_2, rng=rng, n_iters=arranging_iters
    )

    true_distance = sphere_gw_distance(d_1 - 1, d_2 - 1)
    pot_estimated_distance = get_empirical_gw_pot(pts_1, pts_2, epsilon=-1)
    ott_estimated_distance = get_empirical_gw_ott(pts_1, pts_2, epsilon=epsilon)
    return {
        "true_distance": true_distance,
        "pot_estimate": pot_estimated_distance,
        f"ott_estimate_reg{epsilon}": ott_estimated_distance,
    }


def random_trial(d_1, d_2, n_1, n_2, seed=0, epsilon=1e-2):
    rng = jax.random.PRNGKey(seed)
    pts_1 = point_arrangement.generate_points_on_sphere(d_1, n_1, rng=rng)
    pts_2 = point_arrangement.generate_points_on_sphere(d_2, n_2, rng=rng)

    true_distance = sphere_gw_distance(d_1 - 1, d_2 - 1)
    pot_estimated_distance = get_empirical_gw_pot(pts_1, pts_2, epsilon=-1)
    ott_estimated_distance = get_empirical_gw_ott(pts_1, pts_2, epsilon=epsilon)
    return {
        "true_distance": true_distance,
        "pot_estimate": pot_estimated_distance,
        f"ott_estimate_reg{epsilon}": ott_estimated_distance,
    }


def voronoi_trial(d_1, d_2, n_1, n_2, seed=0, epsilon=1e-2, n_larger=1_000_000):
    rng = jax.random.PRNGKey(seed)
    pts_1, wghts_1 = point_arrangement.get_voronoi_sphere_pts(
        d_1, n_1, rng=rng, n_larger=n_larger
    )
    pts_2, wghts_1 = point_arrangement.get_voronoi_sphere_pts(
        d_2, n_2, rng=rng, n_larger=n_larger
    )

    true_distance = sphere_gw_distance(d_1 - 1, d_2 - 1)
    pot_estimated_distance = get_empirical_gw_pot(pts_1, pts_2, epsilon=-1)
    ott_estimated_distance = get_empirical_gw_ott(pts_1, pts_2, epsilon=epsilon)
    return {
        "true_distance": true_distance,
        "pot_estimate": pot_estimated_distance,
        f"ott_estimate_reg{epsilon}": ott_estimated_distance,
    }


def benchmarking_run(
    m: int,
    n: int,
    n_trials: int,
    subsampling_strategy: Literal["random", "arrange", "voronoi"],
    data_path: pathlib.Path,
    samples=np.arange(10, 200, 10),
    verbose=False,
):
    progress_bar = tqdm(samples) if verbose else samples
    trial_outcomes = {}
    for s in progress_bar:
        samples_trial_outcomes = {}
        for t in range(n_trials):
            if subsampling_strategy == "random":
                samples_trial_outcomes[t] = random_trial(m + 1, n + 1, s, s, t, 1e-2)
            if subsampling_strategy == "arrange":
                samples_trial_outcomes[t] = arranged_trial(m + 1, n + 1, s, s, t, 1e-2)
            if subsampling_strategy == "voronoi":
                samples_trial_outcomes[t] = voronoi_trial(m + 1, n + 1, s, s, t, 1e-2)

        trial_outcomes[int(s)] = samples_trial_outcomes

    trial_data = {
        "metadata": {
            "sphere_dimension_1": m,
            "sphere_dimension_2": n,
            "n_trials": n_trials,
            "subsampling_strategy": subsampling_strategy,
        },
        "data": trial_outcomes,
    }

    try:
        with open(
            data_path
            / f"{subsampling_strategy}_trials"
            / f"{subsampling_strategy}_trials_n{n_trials}.json",
            "w",
        ) as f:
            json.dump(trial_data, f)
    except:
        print(
            "Desired directory not found, dumping experiment outputs to current location."
        )
        with open(
            f"{subsampling_strategy}_trials_n{n_trials}.json",
            "w",
        ) as f:
            json.dump(trial_data, f)
