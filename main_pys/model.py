import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils


class GNNStack(nn.Module):
    def __init__(self, linear_dim, in_channels, hidden_dim, output_dim, relu_type, task='node'):
        super(GNNStack, self).__init__()
        self.task = task
        self.relu_type = relu_type
        self.convs = nn.ModuleList([self.build_conv_model(linear_dim, in_channels, hidden_dim,True)])
        self.lns = nn.ModuleList([nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)])
        for _ in range(3):
            self.convs.append(self.build_conv_model(linear_dim, hidden_dim, hidden_dim, False))
            self.lns.append(nn.LayerNorm(hidden_dim))

        self.post_mp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.25),
                                     nn.Linear(hidden_dim, output_dim))
        if task not in ['node', 'graph']:
            raise RuntimeError('Unknown task.')

        self.dropout = 0.25
        self.num_layers = 4
       

    def build_conv_model(self, linear_dim, in_channels, hidden_dim, image_flag):
        if image_flag:
            return CustomConv(linear_dim, in_channels, hidden_dim, self.relu_type)
        return pyg_nn.SAGEConv(in_channels, hidden_dim)

    def forward(self, data):
        """
        Input: data -- a torch_geometric.data.Data object with the following attributes:
            x -- node features
            edge_index -- graph connectivity
            batch -- batch assignment
            y -- node labels
        Output:
            F.log_softmax(x, dim=1) -- node score logits, we can do exp() to get probabilities
            """
        x, edge_index, batch, bd_pred = data.x, data.edge_index, data.batch, data.bd_pred
        if data.num_node_features == 0:
            x = torch.ones(data.num_nodes, 1)
        x = self.convs[0](x, bd_pred, edge_index)
        for i in range(1, self.num_layers):
            x = self.convs[i](x, edge_index)
            emb = x
            if self.relu_type!="relu":
                x = F.leaky_relu(x)
            else:
                x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if i != self.num_layers - 1:
                x = self.lns[i](x)

        if self.task == 'graph':
            x = pyg_nn.global_mean_pool(x, batch)

        x = self.post_mp(x)
        return emb, F.log_softmax(x, dim=1)

    def loss(self, pred, label, weights):
        base_loss = F.nll_loss(pred, label, reduction='none')
        weighted_loss = base_loss*weights
        # This combined with log_softmax in forward() is equivalent to cross entropy
        # as stated in https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        return weighted_loss.mean()

class CustomConv(pyg_nn.MessagePassing):
    def __init__(self, linear_dim, in_channels, out_channels,relu_type):
        super(CustomConv, self).__init__(aggr='add')
        self.lin = nn.Linear(linear_dim, out_channels)
        self.lin_self = nn.Linear(linear_dim, out_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=1, padding=0)
        self.conv_self = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=1, padding=0)
        self.relu_type=relu_type

    def forward(self, x, bd_pred, edge_index):
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        flattened_conv = torch.flatten(self.conv_self(x), start_dim=1) # (1, ~)
        if bd_pred.shape[0]>2:
            flattened_conv = torch.hstack([flattened_conv, bd_pred])
            
        if self.relu_type!="relu":
            self_x = F.leaky_relu(flattened_conv)
        else:
            self_x = F.relu(flattened_conv)
        
        self_x = self.lin_self(self_x)

        if self.relu_type!="relu":
            x_neighbors = F.leaky_relu(flattened_conv)
        else:
            x_neighbors = F.relu(flattened_conv)
        x_neighbors = self.lin(x_neighbors)

        self_and_propogated = self_x + self.propagate(edge_index, x=x_neighbors)
        return self_and_propogated

    def message(self, x_j, edge_index, size):
        return x_j

    def update(self, aggr_out):
        return aggr_out