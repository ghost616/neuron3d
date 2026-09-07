"""spatial_hash.py -- uniform 3-D spatial hash (M1 wiring support).

Provides ``SpatialHash3D``: an O(N) build (single pass bucket assignment over a
uniform grid of cell size ``cell_size``) and O(1) amortized neighborhood
queries (a query touches the constant number of cells that overlap the query
sphere -- for uniform density this is a small constant).  Used by
``network.build_network`` for the input->pool (radius R_IN, k nearest) and
pool->pool (radius R_POOL, k nearest per source) wirings of D11/D13.

Pure NumPy (L0).  Deterministic: candidates within a query are returned sorted
by ascending Euclidean distance; ties broken by ascending index.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = ["SpatialHash3D", "euclidean_dist"]


def euclidean_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Euclidean distance between two (N,3)/(M,3) point sets.

    Args:
        a: (N,3) float64 points.
        b: (M,3) float64 points.

    Returns:
        (N,) float64 distances (a-b along each row).
    """
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return np.sqrt(np.einsum("ij,ij->i", d, d))


class SpatialHash3D:
    """Uniform-grid 3-D spatial hash over a static point set.

    Attributes:
        n (int): number of indexed points.
        cell_size (float): grid cell side length.
        origin (np.ndarray): (3,) grid origin (per-axis minimum of the points).
        grid_shape (Tuple[int, int, int]): number of occupied cells per axis.
    """

    def __init__(self, points: np.ndarray, cell_size: float,
                 origin: Optional[np.ndarray] = None) -> None:
        """Build the hash in O(N) over ``points``.

        Args:
            points: (N,3) float64 point coordinates.
            cell_size: positive grid cell side length (``R_POOL`` for the net).
            origin: optional (3,) grid origin; defaults to per-axis point min.

        Raises:
            ValueError: on bad cell_size, non-(N,3) points or non-finite input.
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(
                f"points must be (N,3), got shape {pts.shape}"
            )
        if pts.size and not np.isfinite(pts).all():
            raise ValueError("points contain NaN/Inf values")
        if cell_size <= 0:
            raise ValueError(f"cell_size must be > 0, got {cell_size}")
        self.n = int(pts.shape[0])
        self.cell_size = float(cell_size)
        if self.n == 0:
            self.origin = np.zeros(3, dtype=np.float64)
            self.grid_shape = (0, 0, 0)
            self._cells: Dict[Tuple[int, int, int], Tuple[int, int]] = {}
            self._order = np.zeros(0, dtype=np.int64)
            self._pts = pts
            return
        self._pts = pts
        self.origin = (
            np.asarray(origin, dtype=np.float64).copy()
            if origin is not None else pts.min(axis=0)
        )
        # Integer cell coordinates per axis (0-based relative to origin).
        ci = np.floor((pts - self.origin) / self.cell_size).astype(np.int64)
        kx, ky, kz = ci[:, 0], ci[:, 1], ci[:, 2]
        # Group points per cell: sort by (kx, ky, kz) and find block starts.
        order = np.lexsort((kz, ky, kx))
        skx, sky, skz = kx[order], ky[order], kz[order]
        if self.n > 1:
            changed = np.flatnonzero(
                (skx[1:] != skx[:-1]) | (sky[1:] != sky[:-1]) | (skz[1:] != skz[:-1])
            ) + 1
        else:
            changed = np.zeros(0, dtype=np.int64)
        starts = np.concatenate(([0], changed))
        ends = np.concatenate((changed, [self.n]))
        self._order = order
        cells: Dict[Tuple[int, int, int], Tuple[int, int]] = {}
        grid_max = np.zeros(3, dtype=np.int64)
        for s, e in zip(starts, ends):
            key = (int(skx[s]), int(sky[s]), int(skz[s]))
            cells[key] = (int(s), int(e))
            grid_max = np.maximum(
                grid_max, np.asarray(key, dtype=np.int64)
            )
        self._cells = cells
        self.grid_shape = tuple(int(v) for v in (grid_max + 1))

    # ------------------------------------------------------------------
    @property
    def points(self) -> np.ndarray:
        """(N,3) float64 indexed points (read-only view)."""
        return self._pts

    # ------------------------------------------------------------------
    def _cell_range(self, point: np.ndarray, radius: float
                    ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """Integer cell ranges [lo, hi] per axis covering the query sphere.

        Args:
            point: (3,) query coordinates.
            radius: query radius (>= 0).

        Returns:
            (x_range, y_range, z_range), each an inclusive (lo, hi) pair.
        """
        c = (np.asarray(point, dtype=np.float64) - self.origin) / self.cell_size
        ext = int(np.ceil(radius / self.cell_size))
        lo = np.floor(c).astype(np.int64) - ext
        hi = np.ceil(c).astype(np.int64) + ext
        return (int(lo[0]), int(hi[0])), (int(lo[1]), int(hi[1])), (int(lo[2]), int(hi[2]))

    def _gather(self, point: np.ndarray, radius: float) -> np.ndarray:
        """Candidate global indices from all cells overlapping the sphere.

        Args:
            point: (3,) query coordinates.
            radius: query radius.

        Returns:
            1-D int64 indices (unordered, within-cell stable).  Empty allowed.
        """
        if self.n == 0:
            return np.zeros(0, dtype=np.int64)
        (x0, x1), (y0, y1), (z0, z1) = self._cell_range(point, radius)
        parts: list[np.ndarray] = []
        for ix in range(x0, x1 + 1):
            for iy in range(y0, y1 + 1):
                for iz in range(z0, z1 + 1):
                    sl = self._cells.get((ix, iy, iz))
                    if sl is not None:
                        parts.append(self._order[sl[0]:sl[1]])
        if not parts:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate(parts)

    # ------------------------------------------------------------------
    def query_radius(self, point: Sequence[float], radius: float,
                     exclude_self: bool = False, self_id: Optional[int] = None
                     ) -> np.ndarray:
        """All indexed points within ``radius`` of ``point``.

        O(1) amortized: only cells overlapping the sphere are scanned.

        Args:
            point: (3,) query coordinates (in the same frame as the points).
            radius: query radius (>= 0).
            exclude_self: if True drop the candidate whose index equals
                ``self_id`` (used to skip the neuron itself for pool->pool).
            self_id: index to drop when ``exclude_self`` is True (required then).

        Returns:
            1-D int64 indices sorted by ascending distance (ties by index);
            empty when no candidate is inside the radius.
        """
        cand = self._gather(point, radius)
        if cand.size == 0:
            return cand
        d = euclidean_dist(self._pts[cand], point)
        inside = d <= radius
        cand, d = cand[inside], d[inside]
        if exclude_self:
            if self_id is None:
                raise ValueError("exclude_self=True requires self_id")
            keep = cand != int(self_id)
            cand, d = cand[keep], d[keep]
        if cand.size < 2:
            return cand
        order = np.lexsort((cand, d))  # distance asc, then index asc
        return cand[order]

    # ------------------------------------------------------------------
    def nearest_k(self, point: Sequence[float], k: int, radius: float,
                  exclude_self: bool = False, self_id: Optional[int] = None
                  ) -> Tuple[np.ndarray, np.ndarray]:
        """The up-to-``k`` nearest indexed points within ``radius``.

        Args:
            point: (3,) query coordinates.
            k: requested number of neighbors (>= 0).
            radius: hard search radius (>= 0); farther candidates are dropped.
            exclude_self: drop candidate index ``self_id`` (pool->pool self).
            self_id: index to drop when ``exclude_self`` is True.

        Returns:
            (indices, distances): 1-D int64 indices sorted by ascending
            distance and 1-D float64 distances; both of length
            ``min(k, available-within-radius)`` (fewer than k is allowed when
            the radius is too small / domain boundary is hit).
        """
        idx = self.query_radius(point, radius, exclude_self=exclude_self,
                                self_id=self_id)
        if idx.size <= k:
            if idx.size == 0:
                return idx, np.zeros(0, dtype=np.float64)
            return idx, euclidean_dist(self._pts[idx], point)
        top = idx[:k]
        return top, euclidean_dist(self._pts[top], point)