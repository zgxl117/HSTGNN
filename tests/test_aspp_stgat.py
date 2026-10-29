import pytest

torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")
HeteroData = pytest.importorskip("torch_geometric.data").HeteroData

from models.models import ASPP_STGATs


def make_toy_heterodata(
    num_ff=4,
    num_modules=2,
    time_window_size=8,
    edge_feat_size=2,
):
    data = HeteroData()

    data["ff"].x = torch.randn(num_ff, time_window_size)
    data["ff"].y = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    data["ff"].node_ids = torch.arange(num_ff)
    data["ff"].train_mask = torch.tensor([True, True, False, False])
    data["ff"].val_mask = torch.tensor([False, False, True, False])
    data["ff"].test_mask = torch.tensor([False, False, False, True])

    data["module"].x = torch.randn(num_modules, time_window_size)

    data["ff", "connects", "ff"].edge_index = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 0],
        ],
        dtype=torch.long,
    )
    data["ff", "connects", "ff"].edge_attr = torch.randn(4, edge_feat_size)

    data["ff", "belongs_to", "module"].edge_index = torch.tensor(
        [
            [0, 1, 2, 3],
            [0, 0, 1, 1],
        ],
        dtype=torch.long,
    )
    data["module", "contains", "ff"].edge_index = torch.tensor(
        [
            [0, 0, 1, 1],
            [0, 1, 2, 3],
        ],
        dtype=torch.long,
    )

    return data


def test_aspp_stgat_forward_shape_and_backward():
    num_classes = 5
    data = make_toy_heterodata()

    model = ASPP_STGATs(
        num_nodes=4,
        time_window_size=8,
        edge_vector_size=4,
        aspp_dilation_list=[1, 2],
        aspp_t_out_size=1,
        aspp_s_hid_size=8,
        s_feat_size=8,
        num_spatio_layers=2,
        edge_hid_size=4,
        gat_hid_heads=1,
        num_classes=num_classes,
    )

    out = model(data)

    assert out.shape == (4, num_classes)
    assert torch.isfinite(out).all()

    mask = data["ff"].train_mask
    loss = F.cross_entropy(out[mask], data["ff"].y[mask])
    loss.backward()

    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
