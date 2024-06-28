from jax import random, numpy as jnp, value_and_grad, jit
import jax
from jax.typing import ArrayLike
from typing import Optional
import numpy as np


def generate_points_on_sphere(
    ambient_dimension: int,
    n_points: int,
    rng: Optional[jax.Array],
) -> jnp.ndarray:
    rng = random.PRNGKey(0) if rng is None else rng
    points = random.normal(rng, shape=(n_points, ambient_dimension))
    return points / jnp.linalg.norm(points, axis=-1, keepdims=True)


def potential(pts: ArrayLike) -> float:
    epsilon = 1e-6
    distances = jnp.sqrt(
        jnp.sum(jnp.square(pts[:, None] - pts[None, :]), axis=-1) + epsilon
    )
    inverse_distances = jnp.triu(1 / (distances + epsilon), 1)
    return jnp.mean(inverse_distances)


def normalize_points(pts: ArrayLike) -> jnp.ndarray:
    return pts / jnp.linalg.norm(pts, axis=-1, keepdims=True)


def repel_points(pts: ArrayLike, lr=0.1) -> jnp.ndarray:
    val, grad = value_and_grad(potential)(pts)
    new_pts = pts - lr * grad
    return normalize_points(new_pts), val


def arrange_sphere_pts(pts: ArrayLike, n_iters: int = 1000):
    vals = []
    pts_hist = []
    repel_points_comp = jit(repel_points)
    for _ in range(n_iters):
        pts, val = repel_points_comp(pts)
        vals.append(val)
        pts_hist.append(pts)
    return pts, vals, jnp.stack(pts_hist)


def get_arranged_sphere_pts(
    ambient_dimension: int,
    n_points: int,
    n_iters: int = 1000,
    rng: Optional[jax.Array] = None,
):
    pts = generate_points_on_sphere(ambient_dimension, n_points, rng)
    pts, vals, pts_hist = arrange_sphere_pts(pts, n_iters)
    return pts, vals, pts_hist


def gaussian_pts(
    dim: int, n_points: int, lim: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    one_d_pts = jnp.linspace(-lim, lim, n_points, endpoint=True)
    one_d_wghts = jax.scipy.stats.norm.pdf(one_d_pts)
    pts = jnp.stack(jnp.meshgrid(*[one_d_pts for _ in range(dim)])).reshape(dim, -1)
    mask = (jnp.linalg.norm(pts, axis=0) <= lim).reshape(-1)
    wghts = (
        jnp.stack(jnp.meshgrid(*[one_d_wghts for _ in range(dim)]))
        .reshape(dim, -1)
        .prod(axis=0)
    )
    pts = pts.swapaxes(0, 1)[mask]
    wghts = wghts[mask]

    return pts, (wghts) / wghts.sum()


def get_furthest_point(pts_subset: jnp.ndarray, pts_all: jnp.ndarray) -> jnp.ndarray:
    distances = jnp.sum(
        jnp.square(pts_subset[:, None, :] - pts_all[None, :, :]), axis=-1
    )
    minimal_distances = jnp.min(distances, axis=0)
    furthest_point_idx = jnp.argmax(minimal_distances)
    return pts_all[furthest_point_idx]


def get_voronoi_sphere_pts(
    ambient_dimension: int,
    n_points: int,
    n_larger: int = 1_000_000,
    rng: Optional[jax.Array] = None,
    reweight: bool = True,
):
    rng = random.PRNGKey(0) if rng is None else rng
    pts_larger = generate_points_on_sphere(ambient_dimension, n_larger, rng)
    pts = []
    pt = random.choice(rng, pts_larger)
    pts.append(pt)
    distances = jnp.sum(jnp.square(pt - pts_larger), axis=-1)

    for _ in range(n_points - 1):
        pt = pts_larger[jnp.argmax(distances)]
        distances = jnp.minimum(
            distances, jnp.sum(jnp.square(pt - pts_larger), axis=-1)
        )
        pts.append(pt)

    pts = jnp.stack(pts)

    nearest_pts = jnp.sum(
        jnp.square(pts[:, None, :] - pts_larger[None, :, :]), axis=-1
    ).argmin(0)
    wghts = (
        jnp.unique_counts(nearest_pts).counts / n_larger
        if reweight
        else jnp.ones(n_points) / n_points
    )

    return pts, wghts
