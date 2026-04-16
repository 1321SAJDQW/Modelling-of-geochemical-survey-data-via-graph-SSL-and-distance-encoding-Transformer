#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This code performs geochemical anomaly recognition based on a graph self-supervised learning and distance encoding Transformer model with edge weights.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle as pkl
import itertools
import argparse
import warnings
from torch_geometric.nn import GCNConv
from Loss import Contrastive_Loss
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, cohen_kappa_score, \
    matthews_corrcoef

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------------------------------------------------
# Parameter Configuration
# ----------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Distance Encoded GCN-Transformer SSL")
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--hidden_size', type=int, default=256, help='Hidden layer size')
parser.add_argument('--emb_size', type=int, default=512, help='Embedding size')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
parser.add_argument('--mask_rate_geo', type=float, default=0.15, help='Mask rate for geochemical features')
parser.add_argument('--l2', type=float, default=0.0005, help='Weight decay')
parser.add_argument('--lr_step_rate', type=float, default=0.85, help='Learning rate step rate')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--tau', type=float, default=3.0, help='Temperature in the contrastive loss')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------------------------------------------------------------------------------------------------
# Load Data and Extract Graph Topology
# ----------------------------------------------------------------------------------------------------------------------

with open('graph_1500_weighted.pkl', 'rb') as f:
    graph = pkl.load(f)

data = pd.read_csv("data.csv")
geo_features = data.drop(columns=['FID', 'XX', 'YY', 'Label']).values
geo_features = torch.from_numpy(geo_features).float().to(device)
labels = torch.tensor(data['Label'].values, dtype=torch.long).to(device)

print(f"Geo Feature Dim: {geo_features.shape[1]}")

sparse_adj = graph['adjacency_matrix'].tocoo()
edge_index = torch.tensor(np.vstack((sparse_adj.row, sparse_adj.col)), dtype=torch.long).to(device)
edge_weight = torch.tensor(sparse_adj.data, dtype=torch.float).to(device)


dense_edge_weights = torch.tensor(graph['adjacency_matrix'].toarray(), dtype=torch.float32).to(device)


shortest_path = np.load('shortest_paths_matrix.npy')
shortest_path = torch.tensor(shortest_path, dtype=torch.float32).to(device)
shortest_path_inv = torch.where(shortest_path > 0, 1.0 / shortest_path, torch.zeros_like(shortest_path))


def mask_features(features, mask_rate):
    mask = torch.rand(features.shape).to(device) < mask_rate
    masked_features = features.clone()
    masked_features[mask] = 0.0
    return masked_features


masked_geo_features = mask_features(geo_features, mask_rate=args.mask_rate_geo)


# ----------------------------------------------------------------------------------------------------------------------
# Model Definitions
# ----------------------------------------------------------------------------------------------------------------------

class WeightedGCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(WeightedGCNEncoder, self).__init__()
        self.gc1 = GCNConv(input_dim, hidden_dim)
        self.gc2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_idx, edge_wt):
        # Strict inclusion of edge weights in GCN aggregation
        x = F.relu(self.gc1(x, edge_idx, edge_weight=edge_wt))
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.gc2(x, edge_idx, edge_weight=edge_wt)
        return x


class LinearClassifier(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

    def forward(self, seq):
        ret = self.fc(seq)
        return F.log_softmax(ret, dim=1)


class CustomDistanceTransformerLayer(nn.Module):

    def __init__(self, emb_size, lambda_param=1.0):
        super(CustomDistanceTransformerLayer, self).__init__()
        self.lambda_param = lambda_param
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)

        self.ffn = nn.Sequential(
            nn.Linear(emb_size, emb_size * 4),
            nn.ReLU(),
            nn.Linear(emb_size * 4, emb_size),
            nn.Dropout(args.dropout)
        )

    def forward(self, x, edge_weights, shortest_path_inv):
        norm_x = self.norm1(x)
        d_k = norm_x.size(-1)

        attn_scores = torch.matmul(norm_x, norm_x.transpose(-2, -1)) / (d_k ** 0.5)
        attn_scores = attn_scores + self.lambda_param * shortest_path_inv
        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_weights, norm_x)

        x = x + attn_output

        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output

        return x


class DistanceEncodedTransformer(nn.Module):
    def __init__(self, emb_size, num_layers=4, lambda_param=1.0):
        super(DistanceEncodedTransformer, self).__init__()
        self.layers = nn.ModuleList([
            CustomDistanceTransformerLayer(emb_size, lambda_param) for _ in range(num_layers)
        ])
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x, edge_weights, shortest_path_inv):
        for layer in self.layers:
            x = layer(x, edge_weights, shortest_path_inv)
        return self.layernorm(x)


# ----------------------------------------------------------------------------------------------------------------------
# Prepare Models and Optimizers
# ----------------------------------------------------------------------------------------------------------------------

GCN = WeightedGCNEncoder(geo_features.shape[1], args.hidden_size, args.emb_size).to(device)
Transformer = DistanceEncodedTransformer(emb_size=args.emb_size).to(device)
Classifier = LinearClassifier(args.emb_size, 2).to(device)

optimizer_encoder = torch.optim.Adam(itertools.chain(GCN.parameters(), Transformer.parameters()), lr=args.lr,
                                     weight_decay=args.l2)
optimizer_classifier = torch.optim.Adam(Classifier.parameters(), lr=0.001, weight_decay=0.01)
scheduler_encoder = torch.optim.lr_scheduler.StepLR(optimizer_encoder, step_size=10, gamma=args.lr_step_rate)

# ----------------------------------------------------------------------------------------------------------------------
# Self-Supervised Pre-training
# ----------------------------------------------------------------------------------------------------------------------

for epoch in range(args.epochs):
    GCN.train()
    Transformer.train()
    optimizer_encoder.zero_grad()

    emb_before_gcn = GCN(geo_features, edge_index, edge_weight)
    emb_before_final = Transformer(emb_before_gcn, dense_edge_weights, shortest_path_inv)

    emb_after_gcn = GCN(masked_geo_features, edge_index, edge_weight)
    emb_after_final = Transformer(emb_after_gcn, dense_edge_weights, shortest_path_inv)

    loss = Contrastive_Loss(emb_before_final, emb_after_final, tau=args.tau)

    loss.backward()
    optimizer_encoder.step()
    scheduler_encoder.step()

    print(f'Epoch {epoch:03d}: Contrastive Loss = {loss.item():.4f}')

# ----------------------------------------------------------------------------------------------------------------------
# Global Classifier Training
# ----------------------------------------------------------------------------------------------------------------------

GCN.eval()
Transformer.eval()

with torch.no_grad():
    emb_gcn = GCN(geo_features, edge_index, edge_weight)
    emb = Transformer(emb_gcn, dense_edge_weights, shortest_path_inv)

label_arr = labels.cpu().numpy()
pos_idx = np.where(label_arr == 1)[0]
neg_idx = np.where(label_arr == 0)[0]

train_pos_idx = np.random.choice(pos_idx, int(0.8 * pos_idx.shape[0]), replace=False)
test_pos_idx = np.setdiff1d(pos_idx, train_pos_idx)
train_neg_idx = np.random.choice(neg_idx, int(0.8 * neg_idx.shape[0]), replace=False)
test_neg_idx = np.setdiff1d(neg_idx, train_neg_idx)

train_idx = np.append(train_pos_idx, train_neg_idx)
test_idx = np.append(test_pos_idx, test_neg_idx)

Classifier = nn.Sequential(
    nn.Linear(emb.shape[1], 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 2),
    nn.LogSoftmax(dim=1)
).to(device)

optimizer_classifier = torch.optim.Adam(Classifier.parameters(), lr=0.001, weight_decay=0.01)

for i in range(100):
    global_classifier = Classifier
    global_classifier.train()
    optimizer_classifier.zero_grad()

    y_pred = global_classifier(emb)
    classifier_loss = F.nll_loss(y_pred[train_idx], labels[train_idx])

    classifier_loss.backward()
    optimizer_classifier.step()

global_classifier.eval()
with torch.no_grad():
    out = global_classifier(emb)
    probability = torch.exp(out)
    result = probability[:, 1].cpu().numpy()

data['Prediction_Probability'] = result
data.to_csv('Prediction_GCN-Transformer_DE.csv', index=False)

# ----------------------------------------------------------------------------------------------------------------------
# Leave-One-Out Cross-Validation
# ----------------------------------------------------------------------------------------------------------------------

# y_true_all, y_pred_prob_all, y_pred_class_all = [], [], []

# for i in range(len(pos_idx)):
#     cv_test_pos_idx = np.array([pos_idx[i]])
#     cv_train_pos_idx = np.delete(pos_idx, i)

#     cv_test_neg_idx = np.array([neg_idx[i % len(neg_idx)]])
#     cv_train_neg_idx = np.delete(neg_idx, i % len(neg_idx))

#     cv_train_idx = np.concatenate([cv_train_pos_idx, cv_train_neg_idx])
#     cv_test_idx = np.concatenate([cv_test_pos_idx, cv_test_neg_idx])

#     cv_classifier = LinearClassifier(args.emb_size, 2).to(device)
#     cv_optimizer = torch.optim.Adam(cv_classifier.parameters(), lr=0.001, weight_decay=0.01)

#     for epoch in range(100):
#         cv_classifier.train()
#         cv_optimizer.zero_grad()
#         cv_out = cv_classifier(emb)
#         cv_loss = F.nll_loss(cv_out[cv_train_idx], labels[cv_train_idx])
#         cv_loss.backward()
#         cv_optimizer.step()

#     cv_classifier.eval()
#     with torch.no_grad():
#         cv_test_out = cv_classifier(emb)
#         cv_prob = torch.exp(cv_test_out)

#         y_true_all.extend(label_arr[cv_test_idx])
#         y_pred_prob_all.extend(cv_prob[cv_test_idx, 1].cpu().numpy())
#         y_pred_class_all.extend(cv_prob[cv_test_idx].argmax(dim=1).cpu().numpy())

# precision = precision_score(y_true_all, y_pred_class_all, zero_division=0)
# recall = recall_score(y_true_all, y_pred_class_all, zero_division=0)
# f1 = f1_score(y_true_all, y_pred_class_all, zero_division=0)
# auc = roc_auc_score(y_true_all, y_pred_prob_all)
# acc = accuracy_score(y_true_all, y_pred_class_all)
# kappa = cohen_kappa_score(y_true_all, y_pred_class_all)
# mcc = matthews_corrcoef(y_true_all, y_pred_class_all)

# # ----------------------------------------------------------------------------------------------------------------------
# # Output Results
# # ----------------------------------------------------------------------------------------------------------------------

# print("\n" + "=" * 75)
# print(f"{'Sample_ID':<20} | {'True_Label':<12} | {'Pred_Class':<10} | {'Probability':<12}")
# print("-" * 75)

# all_sample_names, all_probs, all_true_labels = [], [], []

# for i in range(0, len(y_pred_prob_all), 2):
#     pos_id = f"Deposit_{i // 2 + 1}"
#     pos_prob = y_pred_prob_all[i]
#     pos_pred = y_pred_class_all[i]
#     print(f"{pos_id:<20} | {1:<12} | {pos_pred:<10} | {pos_prob:.4f}")

#     all_sample_names.append(pos_id)
#     all_probs.append(pos_prob)
#     all_true_labels.append(1)

#     neg_id = f"Non_Deposit_{i // 2 + 1}"
#     neg_prob = y_pred_prob_all[i + 1]
#     neg_pred = y_pred_class_all[i + 1]
#     print(f"{neg_id:<20} | {0:<12} | {neg_pred:<10} | {neg_prob:.4f}")

#     all_sample_names.append(neg_id)
#     all_probs.append(neg_prob)
#     all_true_labels.append(0)

# print("\n" + "=" * 75)
# print("LOOCV Final Evaluation Metrics (Based on All 22 Samples)")
# print("-" * 75)
# metrics = {
#     'Precision': precision, 'Recall': recall, 'F1-Score': f1,
#     'AUC': auc, 'ACC': acc, 'Kappa': kappa, 'MCC': mcc
# }
# for name, value in metrics.items():
#     print(f"{name:<15}: {value:.4f}")
# print("=" * 75 + "\n")

# full_results_df = pd.DataFrame({'Sample': all_sample_names, 'True_Label': all_true_labels, 'Prob': all_probs})
# full_results_df.to_csv('DE_Transformer_SSL_22_Samples.csv', index=False)
# pd.DataFrame({'Metric': list(metrics.keys()), 'Score': list(metrics.values())}).to_csv(
#     'DE_Transformer_SSL_metrics.csv', index=False)
