# graphsage_baseline.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

try:
    # torch_scatter 常见于 PyG 环境
    from torch_scatter import scatter_mean
except Exception:
    scatter_mean = None


class TemporalEncoder(nn.Module):
    """
    把每个节点的时间窗特征编码成一个定长 embedding。
    兼容输入：
      - x: (N, T)
      - x: (N, T, C)
      - x: (N, T*C)  (如果你知道 C，可在外面 reshape；这里会当成 (N, T*C) 的 1D 序列)
    """
    def __init__(self, out_dim: int, in_channels: int = 1, hidden_dim: int = 64, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding)
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> (N, C, L)
        if x.dim() == 2:
            # (N, T) -> (N, 1, T)
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            # (N, T, C) -> (N, C, T)
            x = x.permute(0, 2, 1).contiguous()
        else:
            raise ValueError(f"Unsupported x shape: {tuple(x.shape)}")

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # 全局池化：把时间维 L 压缩成 1
        x = x.mean(dim=-1)  # (N, hidden_dim)
        x = self.proj(x)    # (N, out_dim)
        return x


class EdgeAttrToNode(nn.Module):
    """
    GraphSAGE 不直接用 edge_attr，这里提供一个“把边特征聚合到节点”的轻量适配：
    - 先把 edge_attr -> edge_emb
    - 然后按 dst 节点做 mean 聚合，得到每个节点的 edge-context 向量
    """
    def __init__(self, edge_in_dim: int, edge_emb_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(edge_in_dim, edge_emb_dim),
            nn.ReLU(),
            nn.Linear(edge_emb_dim, edge_emb_dim),
        )

    def forward(self, edge_index: torch.Tensor, edge_attr: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if scatter_mean is None:
            # 环境没有 torch_scatter 时，直接返回 0（保证能跑）
            return edge_attr.new_zeros((num_nodes, 0))

        src, dst = edge_index
        e = self.mlp(edge_attr)  # (E, edge_emb_dim)
        agg = scatter_mean(e, dst, dim=0, dim_size=num_nodes)  # (N, edge_emb_dim)
        return agg


class GraphSAGEBaseline(nn.Module):
    """
    时间窗编码(每节点) + GraphSAGE(空间传播) + 节点级分类
    """
    def __init__(
        self,
        num_classes: int = 2,
        node_emb_dim: int = 128,
        sage_hidden_dim: int = 128,
        num_sage_layers: int = 3,
        dropout: float = 0.2,
        # edge_attr 适配（可选）
        use_edge_attr: bool = False,
        edge_in_dim: int = 0,
        edge_emb_dim: int = 32,
        # 如果你的 x 是 (N,T,C)，这里的 in_channels=C；若 x 是 (N,T)，则 in_channels=1
        in_channels: int = 1,
    ):
        super().__init__()
        self.dropout = dropout
        self.use_edge_attr = use_edge_attr and (edge_in_dim > 0)

        self.temporal = TemporalEncoder(out_dim=node_emb_dim, in_channels=in_channels)

        if self.use_edge_attr:
            self.edge2node = EdgeAttrToNode(edge_in_dim=edge_in_dim, edge_emb_dim=edge_emb_dim)
            sage_in_dim = node_emb_dim + edge_emb_dim
        else:
            self.edge2node = None
            sage_in_dim = node_emb_dim

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(sage_in_dim, sage_hidden_dim))
        for _ in range(num_sage_layers - 1):
            self.convs.append(SAGEConv(sage_hidden_dim, sage_hidden_dim))

        self.classifier = nn.Linear(sage_hidden_dim, num_classes)

    def forward(self, data) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index
        num_nodes = data.num_nodes if hasattr(data, "num_nodes") and data.num_nodes is not None else x.size(0)

        h = self.temporal(x)  # (N, node_emb_dim)

        if self.use_edge_attr and hasattr(data, "edge_attr") and data.edge_attr is not None:
            edge_ctx = self.edge2node(edge_index, data.edge_attr, num_nodes)  # (N, edge_emb_dim) 或 (N,0)
            if edge_ctx.numel() > 0:
                h = torch.cat([h, edge_ctx], dim=-1)

        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        logits = self.classifier(h)  # (N, num_classes)  —— 注意：不做 sigmoid/softmax
        return logits
