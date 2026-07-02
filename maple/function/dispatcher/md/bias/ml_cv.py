"""Committor-based machine-learned collective variable (ML-CV) for MAPLE MD.

A small torch MLP ``q_theta(x)`` maps physical descriptors of a configuration
(internal coordinates -- pairwise distances / dihedrals, the descriptor is
PLUGGABLE) to a scalar in ``[0, 1]`` (sigmoid output = committor-like). It is
trained toward the COMMITTOR via the variational principle and then exposed as a
collective variable ``value(x)`` / ``gradient(x)`` (via torch autograd) so it
plugs into the SAME CV slot that metadynamics / OPES / umbrella already consume
(:mod:`bias.batched_metad`, :mod:`bias.umbrella`) -- no integrator or metaD
change: the learned CV rides the existing ``BatchedNVT._forces_au`` additive-bias
hook exactly like the built-in COM-COM distance CV.

Variational committor training
------------------------------
The committor ``q(x)`` (probability of reaching product B before reactant A) is
the minimizer of the Dirichlet energy weighted by the sampled/Boltzmann density,

    L[q] = INT |grad q(x)|^2 dmu(x)      subject to   q|A = 0 ,  q|B = 1 ,

with ``dmu ~ exp(-beta U)`` (Kang, Zhang & Tiwary, *Nat. Comput. Sci.* 2024,
10.1038/s43588-024-00645-0; Rotskoff & Vanden-Eijnden style variational
committor / neural committor, e.g. Li, Lin & Ren, *J. Chem. Phys.* 2019,
10.1063/1.5110439). Here the integral is a weighted empirical average over a
provided set of configurations (weights = the sampled/Boltzmann density; uniform
if the configs are already drawn from ``mu``); the boundary conditions are added
as quadratic penalties ``lambda (<q^2>_A + <(1-q)^2>_B)``. Optional committor
labels (if given) add a supervised MSE term. The Dirichlet gradient ``grad q`` is
taken by torch autograd (``create_graph=True``) so it is itself differentiable
for the Adam step.

As a collective variable
------------------------
* ``value(pos)``          -> scalar ``q_theta in [0,1]``
* ``gradient(pos)``       -> ``dq_theta/dx``  (n, 3)  via torch autograd
* ``value_and_grad(pos)`` -> ``(q, dq/dx)``  (one autograd pass; the shape/return
  contract MATCHES the built-in :func:`bias.batched_metad._com_distance_cv_grad`)
* ``cvs_and_grads(coord_np, ptr)`` -> ``(cvs (B,), [grad (n,3) ...])`` -- the SAME
  signature :class:`bias.batched_metad.BatchedMetaD` uses to feed the metaD/OPES
  engine, so :class:`CommittorMetaD` (a thin ``BatchedMetaD`` subclass that only
  swaps the CV source) reuses the metaD ``apply`` contract VERBATIM.

Units
-----
``pos`` are Angstrom (n, 3); the CV is dimensionless (``q in [0,1]``) and its
gradient is 1/Angstrom, so the bias force ``F_i = -dV/dq * dq/dx_i`` [Ha/A] is
converted to a.u. (Ha/Bohr) by ``HA_PER_ANG_TO_AU`` exactly as the COM-distance
CV path does. Isolated (non-periodic) replicas (inherited from ``BatchedNVT``).

ponytail shortcuts (name the upgrade path)
------------------------------------------
* Descriptors are simple internal coordinates (pairwise distances / dihedrals).
  UPGRADE (ES-8, IMPLEMENTED): :class:`MLIPLatentDescriptor` swaps in MLIP-latent
  features (the MACE per-atom node embedding, pooled to a fixed-size vector) as the
  descriptor -- the CV/train API is unchanged (it is just another callable
  ``pos->features``). With the positions fed as the mace forward's autograd leaf the
  descriptor is DIFFERENTIABLE-through-positions, so ``dq/dx`` stays EXACT
  end-to-end; a detached mode gives a fixed per-frame descriptor (``dq/dfeat`` only).
* The Dirichlet energy is computed with the IDENTITY metric in descriptor space
  (``|grad_feat q|^2``). For the value/gradient CV used to BIAS we still take the
  exact physical ``dq/dx`` via autograd through the descriptor. UPGRADE: use the
  mass-weighted / Jacobian metric ``grad_x q . M^{-1} . grad_x q`` in training for
  the geometrically exact variational committor.
* save/load persists the net + input normalization + arch; the descriptor spec is
  provided by the caller at load (kept minimal).
"""

import numpy as np

from ..utils import HA_PER_ANG_TO_AU          # Ha/Angstrom -> Ha/Bohr (a.u. force)
from .batched import _ptr_to_np               # CUDA-aware _ptr coercion (reuse)
from .batched_metad import BatchedMetaD       # reuse the metaD apply/deposit contract


# --------------------------------------------------------------------------- #
#  Pluggable descriptors: pos (n,3) Angstrom  ->  feature vector (d,), torch-
#  differentiable so autograd propagates dq/d(feat) back to dq/dx. Any object
#  with ``__call__(pos_tensor)->(d,)`` and ``n_features()`` works.
# --------------------------------------------------------------------------- #
class IdentityDescriptor:
    """Features = the flattened input (toy / already-descriptor inputs, e.g. a
    1-D/2-D reaction coordinate). ``n_features`` = ``dim``."""

    def __init__(self, dim):
        self.dim = int(dim)

    def n_features(self):
        return self.dim

    def __call__(self, pos):
        return pos.reshape(-1)


class PairwiseDistanceDescriptor:
    """Features = Euclidean distances for a fixed list of atom-index pairs.

    ``pairs`` : sequence of ``(i, j)`` 0-based atom indices. ``__call__`` takes a
    torch ``pos`` (n, 3) and returns a torch (len(pairs),) vector; autograd then
    gives ``dq/dx`` through the chain rule."""

    def __init__(self, pairs):
        self.pairs = [(int(i), int(j)) for i, j in pairs]
        if not self.pairs:
            raise ValueError("PairwiseDistanceDescriptor needs >= 1 pair.")

    def n_features(self):
        return len(self.pairs)

    def __call__(self, pos):
        import torch
        feats = [torch.linalg.norm(pos[i] - pos[j]) for i, j in self.pairs]
        return torch.stack(feats)


class DihedralDescriptor:
    """Features = dihedral angles (radians) for a list of ``(i, j, k, l)`` quads
    (drop-in with the same interface as the distance descriptor)."""

    def __init__(self, quads):
        self.quads = [(int(i), int(j), int(k), int(l)) for i, j, k, l in quads]
        if not self.quads:
            raise ValueError("DihedralDescriptor needs >= 1 quad.")

    def n_features(self):
        return len(self.quads)

    def __call__(self, pos):
        import torch
        out = []
        for i, j, k, l in self.quads:
            b1 = pos[j] - pos[i]
            b2 = pos[k] - pos[j]
            b3 = pos[l] - pos[k]
            n1 = torch.linalg.cross(b1, b2)
            n2 = torch.linalg.cross(b2, b3)
            m1 = torch.linalg.cross(n1, b2 / torch.linalg.norm(b2))
            x = torch.dot(n1, n2)
            y = torch.dot(m1, n2)
            out.append(torch.atan2(y, x))
        return torch.stack(out)


class CompositeDescriptor:
    """Concatenate several descriptors into one feature vector."""

    def __init__(self, descriptors):
        self.descriptors = list(descriptors)

    def n_features(self):
        return int(sum(d.n_features() for d in self.descriptors))

    def __call__(self, pos):
        import torch
        return torch.cat([d(pos).reshape(-1) for d in self.descriptors])


class MLIPLatentDescriptor:
    """Descriptor = the MLIP backend's OWN learned per-atom node embeddings, pooled
    to a fixed-size CV input (ES-8 -- the upgrade path named in this module's
    docstring). Instead of hand-picked distances / dihedrals, the committor MLP
    rides the MACE message-passing representation of the configuration.

    The backend (a :class:`maple.function.calculator.mace._maceoff_batch_calculator.
    MaceOffBatchCalc`) exposes ``node_features(coord_leaf, ...)`` -> per-atom node
    feats ``(n, F)``; this descriptor pools them over the atoms (or a selected
    ``group``) to a FIXED-size ``(F,)`` vector (independent of n), so it is a
    drop-in for the existing ``pos -> features`` descriptor contract:
    ``CommittorCV(descriptor=MLIPLatentDescriptor(...))`` works UNCHANGED and feeds
    metaD / OPES via :class:`CommittorMetaD`.

    Differentiability (feasibility crux -- stated honestly)
    -------------------------------------------------------
    * ``differentiable=True`` (default): the positions are fed as the autograd LEAF
      of the mace forward, so the node feats -- hence the pooled descriptor -- carry
      grad back to ``x``. The committor gradient ``dq/dx`` is then EXACT end-to-end
      (autograd through the MACE forward), verified FD-vs-autograd in the gate. The
      only non-smooth points are neighbour-list connectivity changes at ``r_max``
      (measure-zero, identical to the MLIP force's own kinks).
    * ``differentiable=False``: node feats are DETACHED (a fixed per-frame
      descriptor). Biasing still works -- ``dq/dfeat`` is exact through the MLP --
      but the descriptor is piecewise-fixed in x (no ``dq/dx`` from the embedding);
      :meth:`CommittorCV.value_and_grad` (which returns ``dq/dx``) is not applicable
      in that mode.

    Rotation/translation invariance: node feats depend only on edge vectors
    (translation-invariant); ``invariants_only=True`` (default) keeps only the l=0
    channels so pooling is rotation-invariant too -- the CV is a proper scalar
    reaction coordinate.

    The backend is PREPARED for this single CV system at construction (B=1). Pass a
    DEDICATED backend instance: ``prepare()`` resets its batch to the CV molecule,
    so do NOT share it with the MD forces backend.
    """

    KIND = "mlip_latent"

    def __init__(self, backend, atoms=None, atomic_numbers=None,
                 init_positions=None, pool="mean", group=None,
                 invariants_only=True, num_layers=None, differentiable=True):
        import torch
        self.backend = backend
        self.pool = str(pool).lower()
        if self.pool not in ("mean", "sum"):
            raise ValueError("MLIPLatentDescriptor pool must be 'mean' or 'sum'.")
        self.invariants_only = bool(invariants_only)
        self.num_layers = num_layers
        self.differentiable = bool(differentiable)
        atoms = self._resolve_atoms(atoms, atomic_numbers, init_positions)
        self.n_atoms = len(atoms)
        backend.prepare([atoms])                          # B=1: THIS CV system
        self.group = (None if group is None
                      else np.asarray(group, dtype=int).reshape(-1))
        # probe the fixed feature width ONCE (architecture-fixed; geometry-invariant).
        with torch.no_grad():
            nf = backend.node_features(invariants_only=self.invariants_only,
                                       num_layers=self.num_layers, detach=True)
        self._nfeat = int(nf.shape[1])

    @staticmethod
    def _resolve_atoms(atoms, atomic_numbers, init_positions):
        """Build the ASE Atoms that PREPAREs the backend for this CV system. Only Z +
        atom count drive the geometry-invariant static cache; per-call the REAL
        positions are fed as the forward leaf, so a spread dummy geometry is fine
        when only ``atomic_numbers`` is given."""
        if atoms is not None:
            return atoms
        if atomic_numbers is None:
            raise ValueError("MLIPLatentDescriptor needs `atoms` or `atomic_numbers`.")
        from ase import Atoms
        Z = np.asarray(atomic_numbers, dtype=int).reshape(-1)
        if init_positions is None:
            init_positions = np.stack(
                [np.arange(len(Z)) * 1.2, np.zeros(len(Z)), np.zeros(len(Z))], axis=1)
        return Atoms(numbers=Z, positions=np.asarray(init_positions, dtype=float))

    def n_features(self):
        return self._nfeat

    def __call__(self, pos):
        """``pos`` (n,3) Angstrom -> pooled latent feature vector ``(F,)`` (torch).
        Differentiable through the MACE forward w.r.t. ``pos`` (``differentiable=
        True``), so :meth:`CommittorCV.value_and_grad` gets the exact end-to-end
        ``dq/dx``; detached (fixed descriptor) otherwise."""
        import torch
        p = (pos if torch.is_tensor(pos)
             else torch.as_tensor(np.asarray(pos, dtype=float), dtype=torch.float64))
        p = p.reshape(self.n_atoms, 3)
        leaf = p.to(self.backend.device, self.backend.dtype)   # forward leaf (grad kept)
        nf = self.backend.node_features(coord_leaf=leaf,
                                        invariants_only=self.invariants_only,
                                        num_layers=self.num_layers,
                                        detach=not self.differentiable)   # (n, F)
        if self.group is not None:
            idx = torch.as_tensor(self.group, dtype=torch.long, device=nf.device)
            nf = nf.index_select(0, idx)
        feat = nf.mean(0) if self.pool == "mean" else nf.sum(0)           # (F,)
        return feat.to(device=p.device, dtype=torch.float64)             # CV-MLP space


# --------------------------------------------------------------------------- #
#  MLP (built lazily so this module imports without torch: the ``bias`` package
#  stays importable on a torch-less box; torch is pulled only on first real use).
# --------------------------------------------------------------------------- #
def _build_net(in_dim, hidden=(32, 32), seed=0):
    """Small MLP producing a SCALAR LOGIT (sigmoid applied by the CV -> q in
    [0,1]). Tanh activations (smooth C-inf: good for the |grad q|^2 Dirichlet
    training and the autograd/FD gradient gate). float64 (dtype-safe eval)."""
    import torch
    from torch import nn
    torch.manual_seed(int(seed))                       # deterministic init
    layers, prev = [], int(in_dim)
    for h in hidden:
        layers += [nn.Linear(prev, int(h)), nn.Tanh()]
        prev = int(h)
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers).double()


# --------------------------------------------------------------------------- #
#  The learned committor collective variable.
# --------------------------------------------------------------------------- #
class CommittorCV:
    """Learned committor CV ``q_theta(x) in [0,1]`` (train + value/gradient +
    save/load). See the module docstring for the variational training and the CV
    contract it exposes to metaD/OPES/umbrella."""

    KIND = "committor"

    def __init__(self, descriptor, net=None, hidden=(32, 32), seed=0,
                 feature_mean=None, feature_std=None):
        self.descriptor = descriptor
        self.in_dim = int(descriptor.n_features())
        self.hidden = tuple(int(h) for h in hidden)
        self.seed = int(seed)
        self.net = net if net is not None else _build_net(self.in_dim, self.hidden, seed)
        # input normalization (set during train; None => identity).
        self.feature_mean = None if feature_mean is None else np.asarray(feature_mean, float)
        self.feature_std = None if feature_std is None else np.asarray(feature_std, float)

    # ---------------------------------------------------------------- internals
    def _norm(self, feats):
        """Affine input normalization (differentiable; constants have no grad)."""
        if self.feature_mean is None:
            return feats
        import torch
        mean = torch.as_tensor(self.feature_mean, dtype=feats.dtype, device=feats.device)
        std = torch.as_tensor(self.feature_std, dtype=feats.dtype, device=feats.device)
        return (feats - mean) / std

    def _q_from_norm(self, Xn):
        """Normalized features (M, d) -> q (M,) in [0,1] (sigmoid of the logit)."""
        import torch
        return torch.sigmoid(self.net(Xn).squeeze(-1))

    # -------------------------------------------------------------- CV contract
    def value(self, pos):
        """q_theta(pos) in [0,1]. ``pos`` (n,3) Angstrom (or a feature-space point
        for the identity descriptor)."""
        import torch
        with torch.no_grad():
            p = torch.as_tensor(np.asarray(pos, float), dtype=torch.float64)
            feats = self._norm(self.descriptor(p).reshape(-1)).unsqueeze(0)   # (1,d)
            return float(self._q_from_norm(feats)[0])

    def value_and_grad(self, pos):
        """Return ``(q_float, dq/dx (n,3) numpy)`` -- the SAME (scalar, grad)
        contract as :func:`bias.batched_metad._com_distance_cv_grad`. dq/dx is the
        exact physical gradient via torch autograd through the descriptor."""
        import torch
        p = torch.tensor(np.asarray(pos, float), dtype=torch.float64,
                         requires_grad=True)
        feats = self._norm(self.descriptor(p).reshape(-1)).unsqueeze(0)        # (1,d)
        q = self._q_from_norm(feats)[0]                                        # scalar
        (g,) = torch.autograd.grad(q, p, create_graph=False)
        return float(q.detach()), g.detach().cpu().numpy()

    def gradient(self, pos):
        """dq_theta/dx (n,3) via autograd."""
        return self.value_and_grad(pos)[1]

    def cvs_and_grads(self, coord_np, ptr):
        """Per-walker ``(cvs (B,), [grad (n,3) ...])`` from master coords -- the
        SAME signature :class:`BatchedMetaD.cvs_and_grads` exposes, so the learned
        CV is a drop-in for the metaD/OPES engine consumption."""
        ptr = _ptr_to_np(ptr)
        B = len(ptr) - 1
        cvs = np.empty(B, dtype=np.float64)
        grads = []
        for b in range(B):
            pos = np.asarray(coord_np[ptr[b]:ptr[b + 1]], dtype=np.float64)
            q, g = self.value_and_grad(pos)
            cvs[b] = q
            grads.append(g)
        return cvs, grads

    def q_of_features(self, features):
        """Batched q over a (M, d) feature array (no autograd) -> (M,) numpy.
        Convenience for evaluation / toy-committor comparison."""
        import torch
        with torch.no_grad():
            X = torch.as_tensor(np.asarray(features, float), dtype=torch.float64)
            if X.ndim == 1:
                X = X.reshape(-1, 1) if self.in_dim == 1 else X.reshape(1, -1)
            return self._q_from_norm(self._norm(X)).cpu().numpy()

    # ------------------------------------------------------------------ training
    def train(self, features, A_mask, B_mask, weights=None, labels=None,
              epochs=2000, lr=1e-3, boundary_weight=100.0, label_weight=0.0,
              set_norm=True, verbose=False):
        """Fit ``q_theta`` toward the committor via the variational principle.

        Parameters
        ----------
        features : (N, d) array
            Descriptor features of the training configurations (already computed;
            for a molecular CV, ``[descriptor(pos_i) for i]``).
        A_mask, B_mask : (N,) bool
            Reactant-state (q=0) and product-state (q=1) membership.
        weights : (N,) array, optional
            Sampled/Boltzmann density weight of each config for the Dirichlet
            integral ``INT |grad q|^2 dmu`` (default uniform -- correct when the
            configs are already drawn from ``mu``).
        labels : (N,) array, optional
            Known committor values (NaN => unlabeled); adds a supervised MSE term
            weighted by ``label_weight``.
        boundary_weight : float
            Penalty ``lambda`` on ``<q^2>_A + <(1-q)^2>_B``.
        """
        import torch
        X = torch.as_tensor(np.asarray(features, float), dtype=torch.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        N, d = X.shape
        if d != self.in_dim:
            raise ValueError(f"feature dim {d} != CV in_dim {self.in_dim}")
        A = torch.as_tensor(np.asarray(A_mask, bool))
        Bm = torch.as_tensor(np.asarray(B_mask, bool))
        if int(A.sum()) == 0 or int(Bm.sum()) == 0:
            raise ValueError("train needs >= 1 config in each of A and B.")
        w = (torch.ones(N, dtype=torch.float64) if weights is None
             else torch.as_tensor(np.asarray(weights, float), dtype=torch.float64))
        w = w / w.sum()

        if set_norm:
            self.feature_mean = X.mean(0).cpu().numpy()
            self.feature_std = (X.std(0) + 1e-8).cpu().numpy()

        if labels is not None:
            L = torch.as_tensor(np.asarray(labels, float), dtype=torch.float64)
            lab_ok = ~torch.isnan(L)
        else:
            L, lab_ok = None, None

        Xr = X.clone().requires_grad_(True)                    # coords for grad q
        opt = torch.optim.Adam(self.net.parameters(), lr=float(lr))
        for ep in range(int(epochs)):
            opt.zero_grad()
            q = self._q_from_norm(self._norm(Xr))              # (N,)
            (grad,) = torch.autograd.grad(q.sum(), Xr, create_graph=True)  # (N,d)
            dirichlet = (w * (grad ** 2).sum(1)).sum()         # weighted <|grad q|^2>
            bc = (q[A] ** 2).mean() + ((1.0 - q[Bm]) ** 2).mean()
            loss = dirichlet + float(boundary_weight) * bc
            if L is not None and label_weight > 0 and int(lab_ok.sum()) > 0:
                loss = loss + float(label_weight) * ((q[lab_ok] - L[lab_ok]) ** 2).mean()
            loss.backward()
            opt.step()
            if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
                print(f"  [committor train] ep {ep:5d}  loss={float(loss):.5e}  "
                      f"dirichlet={float(dirichlet):.5e}  bc={float(bc):.5e}")
        return self

    # -------------------------------------------------------------- save / load
    def save(self, path):
        """Persist net weights + normalization + arch (torch archive)."""
        import torch
        torch.save({"state_dict": self.net.state_dict(),
                    "in_dim": self.in_dim, "hidden": list(self.hidden),
                    "seed": self.seed,
                    "feature_mean": self.feature_mean,
                    "feature_std": self.feature_std}, path)

    @classmethod
    def load(cls, path, descriptor):
        """Load a saved CV; ``descriptor`` (same spec used at save) is supplied by
        the caller (kept minimal -- the descriptor is not serialized)."""
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cv = cls(descriptor, hidden=tuple(ck["hidden"]), seed=int(ck.get("seed", 0)),
                 feature_mean=ck.get("feature_mean"), feature_std=ck.get("feature_std"))
        cv.net.load_state_dict(ck["state_dict"])
        cv.net.double()
        return cv


# --------------------------------------------------------------------------- #
#  CV -> metaD parity: reuse BatchedMetaD's apply/deposit VERBATIM, swap only the
#  CV source (override cvs_and_grads). This is how the learned CV plugs into the
#  EXISTING metaD/OPES slot -- BatchedMetaD.apply, the engine consumption, the
#  force injection (F_i = -dV/dq * dq/dx_i, a.u.), the hill deposit, the CV/bias
#  logging are all inherited unchanged (batched_metad.py is NOT modified).
# --------------------------------------------------------------------------- #
class CommittorMetaD(BatchedMetaD):
    """Well-tempered metaD / OPES biased along the learned committor CV.

    Same bias contract as :class:`bias.batched_metad.BatchedMetaD`
    (``apply(E, F, calc) -> (E, F)``; attach as ``BatchedNVT._bias``); it just
    reads the CV from a :class:`CommittorCV` instead of the built-in COM-distance
    CV. Engine grid should span the committor range ``cv_min=0, cv_max=1``."""

    KIND = "committor_metad"

    def __init__(self, atoms_list, cv, engine, pace, deposit=True):
        self.B = len(atoms_list)
        if self.B == 0:
            raise ValueError("CommittorMetaD needs >= 1 walker.")
        self.cv = cv
        self.engine = engine
        self.pace = int(pace)
        self.deposit = bool(deposit)
        self._napply = 0
        self._n = [len(at) for at in atoms_list]
        # per-walker logs (mirrors BatchedMetaD; consumed by apply()).
        self.cv_history = [[] for _ in range(self.B)]
        self.bias_history = [[] for _ in range(self.B)]
        self.deposit_steps = []

    # only the CV source changes; everything else is inherited from BatchedMetaD.
    def cvs_and_grads(self, coord_np, ptr):
        ptr = _ptr_to_np(ptr)
        cvs = np.empty(self.B, dtype=np.float64)
        grads = []
        for b in range(self.B):
            pos = np.asarray(coord_np[ptr[b]:ptr[b + 1]], dtype=np.float64)
            q, g = self.cv.value_and_grad(pos)
            cvs[b] = q
            grads.append(g)
        return cvs, grads
