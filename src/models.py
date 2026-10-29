import torch
import torch.nn.functional as F
from torch.nn import Linear, Embedding
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv
import numpy as np

class GNN(torch.nn.Module):
    """
    GATv2 堆疊，層數由 config.GNN_LAYERS 決定（預設 2，與改動前完全等價）。

    子模組刻意沿用 conv1 / conv2 / ... 的命名，因此 GNN_LAYERS=2 時
    state_dict 的鍵與改動前完全相同，既有 checkpoint 可直接載入；
    層數不符時 load_state_dict 會因形狀不符而明確報錯，不會靜默出錯。

    結構：前 N-1 層為 concat 多頭（輸出 hidden_channels），
          最後一層 concat=False（輸出 out_channels），層間以 ELU 銜接。

    卷積層種類由 config.GNN_TYPE 決定（gatv2 / gcn / sage，見 config.py）；
    gcn、sage 的卷積層不吃邊特徵，維度與層數與 gatv2 相同。
    config.GNN_EDGE_IN_ATTN=False 時 GATv2 不建邊特徵的投影層（edge_dim=None）。
    """

    def __init__(self, in_channels, hidden_channels, out_channels, edge_feature_dim,
                 heads=4, layers=None):
        super().__init__()
        import config
        if layers is None:
            layers = int(getattr(config, 'GNN_LAYERS', 2))
        if layers < 1:
            raise ValueError(f"GNN_LAYERS 至少為 1，收到 {layers}")
        self.n_layers = layers
        self.gnn_type = str(getattr(config, 'GNN_TYPE', 'gatv2'))
        self.use_edge_attr = self.gnn_type == 'gatv2' and bool(getattr(config, 'GNN_EDGE_IN_ATTN', True))
        edge_dim = edge_feature_dim if self.use_edge_attr else None

        def make(inp, out, last):
            if self.gnn_type == 'gcn':
                return GCNConv(inp, out)
            if self.gnn_type == 'sage':
                return SAGEConv(inp, out, aggr='mean')
            if last:
                return GATv2Conv(inp, out, heads=heads, concat=False, edge_dim=edge_dim)
            return GATv2Conv(inp, out // heads, heads=heads, edge_dim=edge_dim)

        for i in range(1, layers):
            inp = in_channels if i == 1 else hidden_channels
            setattr(self, f'conv{i}', make(inp, hidden_channels, last=False))
        last_in = hidden_channels if layers > 1 else in_channels
        setattr(self, f'conv{layers}', make(last_in, out_channels, last=True))

    def forward(self, x, edge_index, edge_attr):
        for i in range(1, self.n_layers + 1):
            conv = getattr(self, f'conv{i}')
            x = conv(x, edge_index, edge_attr) if self.use_edge_attr else conv(x, edge_index)
            if i < self.n_layers:
                x = F.elu(x)
        return x

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, edge_feature_dim):
        super().__init__()
        # edge_lin 未在 forward 中使用，但既有 checkpoint 都含有這一層的權重，保留以維持相容
        self.edge_lin = Linear(edge_feature_dim, 32)
        # 修改輸入維度以包含邊特徵（config.GNN_EDGE_IN_PRED=False 時不接邊特徵）
        import config
        self.use_edge_attr = bool(getattr(config, 'GNN_EDGE_IN_PRED', True))
        self.lin1 = Linear(in_channels * 2 + (edge_feature_dim if self.use_edge_attr else 0), 64)
        self.lin2 = Linear(64, 1)

    def forward(self, x, edge_label_index, edge_label_attr):
        # 取得連結的兩個節點嵌入
        node1 = x[edge_label_index[0]]
        node2 = x[edge_label_index[1]]
        
        # # 處理邊特徵
        # edge_features = self.edge_lin(edge_label_attr)
        # edge_features = F.relu(edge_features)
        
        # 拼接節點特徵和邊特徵
        if self.use_edge_attr:
            combined_features = torch.cat([node1, node2, edge_label_attr], dim=1)
        else:
            combined_features = torch.cat([node1, node2], dim=1)
        
        # 預測連結分數
        x = self.lin1(combined_features)
        x = F.relu(x)
        x = self.lin2(x)
        return x.view(-1)

class Model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_feature_dim, vocab_size, embedding_dim, embedding_matrix):
        super().__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        # self.embedding.weight.data.copy_(embedding_matrix)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = False  # 可選，是否允許微調
        # self.gnn = GNN(in_channels + 300, hidden_channels, out_channels, edge_feature_dim)
        self.gnn = GNN(64 + 64, hidden_channels, out_channels, edge_feature_dim)#64+64
        self.link_predictor = LinkPredictor(out_channels, edge_feature_dim)
        self.dropout = torch.nn.Dropout(0.2)
        # 空間特徵升維
        self.spatial_encoder = torch.nn.Sequential(
            torch.nn.Linear(4, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 64),
            # torch.nn.ReLU()#noRelu
        )
        # 文字嵌入降維
        # self.text_encoder = torch.nn.Linear(768, 64)
        self.text_encoder = torch.nn.Sequential(
            torch.nn.Linear(300, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            # torch.nn.Linear(128, 32),
            torch.nn.Linear(128, 64),#64+64
            # torch.nn.ReLU(),#noRelu
            # torch.nn.Dropout(0.3)
        )

    def forward(self, data):
        # 使用label_id屬性
        label_ids = data.label_id
        # 獲取節點特徵
        node_features = data.x
        # 取得每個節點的文字嵌入
        text_embed = self.embedding(label_ids)

        # forward時
        node_features = self.spatial_encoder(node_features)  # (N, 64) 64+64
        text_embed = self.text_encoder(text_embed)           # (N, 64) 64+64
        
        # 拼接空間特徵與文字嵌入
        x = torch.cat([node_features, text_embed], dim=1)
        x = self.dropout(x)
        x = self.gnn(x, data.edge_index, data.edge_attr)
        return self.link_predictor(x, data.edge_label_index, data.edge_label_attr)

    def get_node_embeddings(self, data):
        """僅計算 GNN 節點特徵 (用於 C 方案評估)"""
        label_ids = data.label_id
        node_features = data.x
        text_embed = self.embedding(label_ids)
        node_features = self.spatial_encoder(node_features)#64+64
        text_embed = self.text_encoder(text_embed)#64+64
        x = torch.cat([node_features, text_embed], dim=1)
        # x = self.dropout(x) # 評估時 dropout 不生效
        x = self.gnn(x, data.edge_index, data.edge_attr)
        return x

    def predict_edges(self, node_embeddings, edge_label_index, edge_label_attr):
        """僅計算連線預測 (用於 C 方案分批評估)"""
        return self.link_predictor(node_embeddings, edge_label_index, edge_label_attr)
