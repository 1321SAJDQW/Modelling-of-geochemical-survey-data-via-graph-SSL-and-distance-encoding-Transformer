#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This code performs geochemical anomaly recognition based on a graph self-supervised learning model with edge weights.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle as pkl
from torch_geometric.nn import GCNConv
from Loss import Contrastive_Loss
import argparse
import scipy.sparse as sp

# -------------------------------------------------------------------------------------------------------------
# Parameter Configuration
# -------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Graph-based Mineral Resource Prediction")

parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--hidden_size', type=int, default=256, help='Hidden layer size')
parser.add_argument('--emb_size', type=int, default=512, help='Embedding size')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
parser.add_argument('--mask_rate_geo', type=float, default=0.15, help='Mask rate for geochemical features')
parser.add_argument('--l2', type=float, default=0.0005, help='Weight decay (L2 loss on parameters)')
parser.add_argument('--lr_step_rate', type=float, default=0.85, help='Learning rate step rate')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--tau', type=float, default=3, help='Temperature in the contrastive loss')
parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples for contrastive loss')

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------------------------------------------------------------------------------------------------
# Load Graph Data
# ----------------------------------------------------------------------------------------------------------------------

with open('graph_1500_weighted.pkl', 'rb') as rf1:
    graph = pkl.load(rf1)

data = pd.read_csv("data.csv")
geo_features = data.drop(columns=['FID', 'XX', 'YY', 'Label'])
geo_features = torch.from_numpy(geo_features.values).float()

labels = torch.tensor(data['Label'].values, dtype=torch.long)

geo_features = geo_features.to(device)
labels = labels.to(device)

print(f"Geo Feature Dim: {geo_features.shape[1]}")

# ----------------------------------------------------------------------------------------------------------------------
# Function to Mask Features
# ----------------------------------------------------------------------------------------------------------------------

def mask_features(features, mask_rate):
    mask = torch.rand(features.shape) < mask_rate
    masked_features = features.clone()
    masked_features[mask] = 0
    return masked_features

masked_geo_features = mask_features(geo_features, mask_rate=args.mask_rate_geo)

# ----------------------------------------------------------------------------------------------------------------------
# Convert Adjacency Matrix from Graph
# ----------------------------------------------------------------------------------------------------------------------

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    coo_tensor = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)
    return coo_tensor.to_sparse_csr()

def adj_calculate(graph):
    adj = graph['adjacency_matrix']
    adj_ori = torch.Tensor(adj.toarray()).to(device)
    adj = adj + sp.eye(adj.shape[0])
    adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    adj_label = adj_ori + torch.eye(adj_ori.shape[0]).to(device)
    pos_weight = (adj_ori.shape[0] ** 2 - adj_ori.sum()) / adj_ori.sum()
    norm = adj_ori.shape[0] ** 2 / (2 * (adj_ori.shape[0] ** 2 - adj_ori.sum()))
    return adj, adj_ori, adj_label, pos_weight, norm

adj, adj_ori, adj_label, pos_weight, norm = adj_calculate(graph)

# ----------------------------------------------------------------------------------------------------------------------
# Define GCN Encoder and Classifier
# ----------------------------------------------------------------------------------------------------------------------

class SimpleGCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGCNEncoder, self).__init__()
        self.gc1 = GCNConv(input_dim, hidden_dim)
        self.gc2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.gc2(x, edge_index)
        return x

class Linear_Classifier(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(Linear_Classifier, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

    def forward(self, seq):
        ret = self.fc(seq)
        return F.log_softmax(ret, dim=1)

# ----------------------------------------------------------------------------------------------------------------------
# Prepare GCN Encoder and Classifier
# ----------------------------------------------------------------------------------------------------------------------

GCN = SimpleGCNEncoder(masked_geo_features.shape[1], args.hidden_size, args.emb_size).to(device)
Classifier = Linear_Classifier(args.emb_size, 2).to(device)

optimizer_encoder = torch.optim.Adam(GCN.parameters(), lr=args.lr, weight_decay=args.l2)
optimizer_classifier = torch.optim.Adam(Classifier.parameters(), lr=0.001, weight_decay=0.01)
scheduler_encoder = torch.optim.lr_scheduler.StepLR(optimizer_encoder, step_size=10, gamma=args.lr_step_rate)

# ----------------------------------------------------------------------------------------------------------------------
# Pretraining with Contrastive Loss
# ----------------------------------------------------------------------------------------------------------------------

for epoch in range(args.epochs):
    GCN.train()

    emb_before = GCN(geo_features, adj)
    emb_after = GCN(masked_geo_features, adj)

    Contrastive_loss = Contrastive_Loss(emb_before, emb_after, tau=args.tau)
    total_loss = Contrastive_loss

    optimizer_encoder.zero_grad()
    total_loss.backward()
    optimizer_encoder.step()
    scheduler_encoder.step()

    torch.cuda.empty_cache()

    print(f'Epoch {epoch}: Total_loss = {total_loss.item():.4f}')

# ----------------------------------------------------------------------------------------------------------------------
# Train Classifier (For Global Prediction Map)
# ----------------------------------------------------------------------------------------------------------------------

GCN.eval()
emb = GCN(geo_features, adj).detach()

label1 = labels.to("cpu").numpy()
positiveIndex = np.where(label1 == 1)[0]
negativeIndex = np.where(label1 == 0)[0]

trainPositiveIndex = np.random.choice(positiveIndex, int(0.8 * positiveIndex.shape[0]), replace=False)
testPositiveIndex = np.setdiff1d(positiveIndex, trainPositiveIndex)
trainNegativeIndex = np.random.choice(negativeIndex, int(0.8 * negativeIndex.shape[0]), replace=False)
testNegativeIndex = np.setdiff1d(negativeIndex, trainNegativeIndex)
trainIndex = np.append(trainPositiveIndex, trainNegativeIndex)
testIndex = np.append(testPositiveIndex, testNegativeIndex)

print("Unique labels:", np.unique(label1))
print("Max label:", label1.max())
selected_labels = labels[trainIndex].cpu().numpy()
print(f"Unique values in labels[trainIndex]: {np.unique(selected_labels)}")

for i in range(100):
    optimizer_classifier.zero_grad()
    y_pred = Classifier(emb)
    classifier_loss = F.nll_loss(y_pred[trainIndex], labels[trainIndex])
    classifier_loss.backward()
    optimizer_classifier.step()
    print(f'Epoch {i}: Loss = {classifier_loss.item():.4f}')

# ----------------------------------------------------------------------------------------------------------------------
# Save Predictions
# ----------------------------------------------------------------------------------------------------------------------

torch.save(Classifier.state_dict(), "classifier_best.pth")
torch.save(GCN.state_dict(), "encoder_best.pth")

out = Classifier(emb.detach())

probability = nn.functional.softmax(out, dim=-1)
result = probability[:, 1].to("cpu").detach().numpy()

data['Prediction_Probability'] = result
data.to_csv('Weighted_GCN_SSL.csv', index=False)

# ----------------------------------------------------------------------------------------------------------------------
# Leave-One-Out Cross-Validation (LOOCV) Implementation
# ----------------------------------------------------------------------------------------------------------------------

y_true_pos = []
y_pred_prob_pos = []
y_pred_class_pos = []

for i in range(len(positiveIndex)):
    test_pos_idx = np.array([positiveIndex[i]])
    train_pos_idx = np.delete(positiveIndex, i)

    test_neg_idx = np.array([negativeIndex[i % len(negativeIndex)]])
    train_neg_idx = np.delete(negativeIndex, i % len(negativeIndex))

    cv_train_idx = np.concatenate([train_pos_idx, train_neg_idx])

    cv_classifier = Linear_Classifier(args.emb_size, 2).to(device)
    cv_optimizer = torch.optim.Adam(cv_classifier.parameters(), lr=0.001, weight_decay=0.01)

    for epoch in range(100):
        cv_classifier.train()
        cv_optimizer.zero_grad()
        cv_out = cv_classifier(emb.detach())
        cv_loss = F.nll_loss(cv_out[cv_train_idx], labels[cv_train_idx])
        cv_loss.backward()
        cv_optimizer.step()

    cv_classifier.eval()
    with torch.no_grad():
        cv_test_out = cv_classifier(emb.detach())
        cv_prob = torch.exp(cv_test_out)

        y_true_pos.extend(label1[test_pos_idx])
        y_pred_prob_pos.extend(cv_prob[test_pos_idx, 1].cpu().numpy())
        y_pred_class_pos.extend(cv_prob[test_pos_idx].argmax(dim=1).cpu().numpy())

# ----------------------------------------------------------------------------------------------------------------------
# Output Results (Positive Samples Only)
# ----------------------------------------------------------------------------------------------------------------------

print("\n" + "="*60)
print(f"{'Sample_ID':<15} | {'True_Label':<12} | {'Pred_Class':<10} | {'Probability':<12}")
print("-" * 60)
for i, prob in enumerate(y_pred_prob_pos):
    sample_id = f"Deposit_{i+1}"
    print(f"{sample_id:<15} | {1:<12} | {y_pred_class_pos[i]:<10} | {prob:.4f}")

print("\n" + "="*60)
recall = sum(1 for pred in y_pred_class_pos if pred == 1) / len(y_pred_class_pos)
mean_prob = sum(y_pred_prob_pos) / len(y_pred_prob_pos)
print("Positive Samples Evaluation Metrics")
print("-" * 60)
print(f"{'Recall':<15}: {recall:.4f}")
print(f"{'Mean Prob':<15}: {mean_prob:.4f}")
print("="*60 + "\n")

pd.DataFrame({'Sample': [f'Deposit_{i+1}' for i in range(len(y_pred_prob_pos))], 'Prob': y_pred_prob_pos}).to_csv('Base_Weighted_SSL_11_Positive_Only.csv', index=False)