# native torch_scatter replacement via torch_geometric.utils.scatter (already installed)
from torch_geometric.utils import scatter as _pyg_scatter
def scatter(src, index, dim=0, out=None, dim_size=None, reduce='sum'):
    r = 'sum' if reduce in ('add', 'sum') else reduce
    return _pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce=r)
def scatter_add(src, index, dim=0, out=None, dim_size=None):
    return _pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce='sum')
def scatter_mean(src, index, dim=0, out=None, dim_size=None):
    return _pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce='mean')
import torch as _torch
def radius_graph(pos, r, batch=None, loop=False, max_num_neighbors=32,
                 flow='source_to_target', **kw):
    # ponytail: brute-force cdist radius graph (small molecules <=~50 atoms);
    # drops the fragile torch_cluster compiled dep. upgrade: torch_cluster for large N.
    N = pos.size(0)
    if batch is None:
        batch = pos.new_zeros(N, dtype=_torch.long)
    d = _torch.cdist(pos, pos)
    adj = (d <= r) & (batch.unsqueeze(0) == batch.unsqueeze(1))
    if not loop:
        adj = adj & ~_torch.eye(N, dtype=_torch.bool, device=pos.device)
    src, dst = [], []
    for i in _torch.nonzero(adj.any(0)).flatten().tolist():
        js = _torch.nonzero(adj[:, i]).flatten()
        if js.numel() > max_num_neighbors:
            dj = d[js, i]
            js = js[_torch.topk(dj, max_num_neighbors, largest=False).indices]
        src.append(js); dst.append(_torch.full_like(js, i))
    if src:
        return _torch.stack([_torch.cat(src), _torch.cat(dst)], dim=0)
    return pos.new_empty((2, 0), dtype=_torch.long)
