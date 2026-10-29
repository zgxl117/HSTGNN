import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, LeakyReLU
from torch_geometric.nn import GATConv, HeteroConv

from models.gnnmodels.aspp_stgcn import ASPP
from models.gnnmodels.stgcn import OutputLayer


class HeteroSpatioGATLayer(nn.Module):
    """
    标准异构空间图注意力层。

    节点类型：
        - ff:      寄存器节点
        - module:  功能模块节点

    边类型：
        - ('ff', 'connects', 'ff'):       FF-FF 电路拓扑边，带 edge_attr
        - ('ff', 'belongs_to', 'module'): FF -> Module 归属边
        - ('module', 'contains', 'ff'):   Module -> FF 回流边
    """
    def __init__(self, c_in, c_out, edge_feat_size=10, edge_hid_size=4, heads=1):
        super(HeteroSpatioGATLayer, self).__init__()

        self.edge_mlp = nn.Sequential(
            Linear(edge_feat_size, edge_hid_size, bias=False),
            LeakyReLU(),
            Linear(edge_hid_size, edge_hid_size, bias=False)
        )

        # HeteroConv 内部按边类型分别调用 GATConv。
        # 注意：bipartite GATConv 必须关闭 add_self_loops。
        self.hetero_conv = HeteroConv({
            ("ff", "connects", "ff"): GATConv(
                c_in,
                c_out,
                heads=heads,
                concat=False,
                edge_dim=edge_hid_size,
                add_self_loops=False,
            ),
            ("ff", "belongs_to", "module"): GATConv(
                (c_in, c_in),
                c_out,
                heads=heads,
                concat=False,
                add_self_loops=False,
            ),
            ("module", "contains", "ff"): GATConv(
                (c_in, c_in),
                c_out,
                heads=heads,
                concat=False,
                add_self_loops=False,
            ),
        }, aggr="sum")

    def forward(self, x_ff, x_module, edge_index_dict, ff_edge_attr):
        """
        x_ff:      [num_ff, c]
        x_module:  [num_module, c]
        """
        edge_attr_dict = {
            ("ff", "connects", "ff"): self.edge_mlp(ff_edge_attr)
        }

        x_dict = {
            "ff": x_ff,
            "module": x_module,
        }

        out_dict = self.hetero_conv(
            x_dict,
            edge_index_dict,
            edge_attr_dict=edge_attr_dict,
        )

        out_ff = out_dict.get("ff", torch.zeros_like(x_ff))
        out_module = out_dict.get("module", torch.zeros_like(x_module))

        out_ff = F.elu(out_ff)
        out_module = F.elu(out_module)
        out_ff = F.dropout(out_ff, p=0.6, training=self.training)
        out_module = F.dropout(out_module, p=0.6, training=self.training)
        return out_ff, out_module


class ASPP_STGAT(nn.Module):
    def __init__(
        self,
        num_nodes,
        time_window_size,
        aspp_dilation_list=[2, 4, 8],
        aspp_t_out_size=1,
        aspp_s_hid_size=16,
        s_feat_size=32,
        num_spatio_layers=3,
        edge_feat_size=10,
        edge_hid_size=6,
        gat_hid_heads=1,
        normalization=True,
        num_classes=5,
        hetero_mode="full",
    ):
        super(ASPP_STGAT, self).__init__()
        self.num_spatio_layers = num_spatio_layers
        self.num_classes = num_classes
        self.s_feat_size = s_feat_size
        valid_hetero_modes = {"ff_only", "module_to_ff", "full"}
        if hetero_mode not in valid_hetero_modes:
            raise ValueError(
                f"Unknown hetero_mode: {hetero_mode}. "
                f"Expected one of {sorted(valid_hetero_modes)}."
            )
        self.hetero_mode = hetero_mode

        self.ASPP = ASPP(
            n=num_nodes,
            c_in=1,
            c_out=s_feat_size,
            c_hid=aspp_s_hid_size,
            dia_list=aspp_dilation_list,
            t_w=time_window_size,
            t_out_size=aspp_t_out_size,
            normalization=normalization
        )

        self.hetero_layers = nn.ModuleList([
            HeteroSpatioGATLayer(
                s_feat_size,
                s_feat_size,
                edge_feat_size,
                edge_hid_size,
                heads=1 if i == num_spatio_layers - 1 else gat_hid_heads
            )
            for i in range(num_spatio_layers)
        ])

        self.output = OutputLayer(
            s_feat_size,
            aspp_t_out_size,
            num_nodes,
            num_classes=num_classes,
            normalization=normalization
        )

    @staticmethod
    def _init_module_hidden(x_ff, ff_to_module_edge_index, num_modules):
        """
        根据 ASPP 后的 FF 隐状态初始化 module 隐状态。
        x_ff: [time, num_ff, hidden]
        return: [time, num_modules, hidden]
        """
        device = x_ff.device
        dtype = x_ff.dtype
        t, _, c = x_ff.shape

        x_module = torch.zeros(t, num_modules, c, device=device, dtype=dtype)
        if ff_to_module_edge_index.numel() == 0:
            return x_module

        ff_index = ff_to_module_edge_index[0]
        module_index = ff_to_module_edge_index[1]

        for i in range(t):
            mapped_ff = x_ff[i, ff_index, :]
            x_module[i].index_add_(0, module_index, mapped_ff)

        counts = torch.bincount(module_index, minlength=num_modules).to(device).float().view(1, num_modules, 1)
        x_module = x_module / counts.clamp(min=1.0)
        return x_module

    def forward(self, data):
        # 只对 FF 节点做最终分类；module 节点作为异构消息传递中的辅助语义节点。
        x = data["ff"].x

        ff_edge_index = data["ff", "connects", "ff"].edge_index
        ff_edge_attr = data["ff", "connects", "ff"].edge_attr
        ff_to_module_edge_index = data["ff", "belongs_to", "module"].edge_index
        module_to_ff_edge_index = data["module", "contains", "ff"].edge_index
        num_modules = data["module"].x.size(0)

        # [num_ff, time] -> [time, num_ff, 1]
        x = x.transpose(0, 1)
        x = x.unsqueeze(-1)

        # ASPP 时间卷积后得到 FF 节点时序隐表示：[time, num_ff, s_feat_size]
        x_ff = self.ASPP(x)

        # 用 FF 隐状态均值初始化 module 隐状态：[time, num_module, s_feat_size]
        x_module = self._init_module_hidden(x_ff, ff_to_module_edge_index, num_modules)

        edge_index_dict = {
            ("ff", "connects", "ff"): ff_edge_index,
        }
        if self.hetero_mode == "full":
            edge_index_dict[("ff", "belongs_to", "module")] = ff_to_module_edge_index
        if self.hetero_mode in {"module_to_ff", "full"}:
            edge_index_dict[("module", "contains", "ff")] = module_to_ff_edge_index

        # 每个时间步分别执行异构空间图注意力传播
        for layer in self.hetero_layers:
            ff_outputs = []
            module_outputs = []
            for t in range(x_ff.shape[0]):
                out_ff, out_module = layer(
                    x_ff[t],
                    x_module[t],
                    edge_index_dict,
                    ff_edge_attr,
                )
                ff_outputs.append(out_ff)
                module_outputs.append(out_module)

            delta_ff = torch.stack(ff_outputs, dim=0)
            delta_module = torch.stack(module_outputs, dim=0)

            # 残差融合：FF-FF 拓扑传播 + FF/Module 异构关系传播
            x_ff = x_ff + delta_ff
            x_module = x_module + delta_module

        # 只输出 FF 节点分类结果：[num_ff, num_classes]
        out = self.output(x_ff)
        return out
