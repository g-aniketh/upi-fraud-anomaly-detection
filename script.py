import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import networkx as nx

# Try to import necessary libraries and provide guidance if they are missing.
try:
    import kagglehub
except ImportError:
    print("KaggleHub not found. Please ensure it is installed: 'pip install kagglehub'")
    exit()

try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv, SAGEConv
    from torch_geometric.explain import Explainer
    from torch_geometric.explain.algorithm import CaptumExplainer

except ImportError:
    print("PyTorch or PyTorch Geometric not found. Please run '!pip install torch_geometric captum networkx' in a Colab cell.")
    exit()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve, auc
)


print("=" * 80)
print("Phase 0: Environment Setup Complete")
if 'torch' in locals():
    print(f"PyTorch version: {torch.__version__}")
print("=" * 80)


################################################################################
# Phase 1: Load and Preprocess Data for Graph Construction
################################################################################
print("Phase 1: Loading and Preprocessing Data...")

try:
    dataset_path_dir = kagglehub.dataset_download("rupakroy/online-payments-fraud-detection-dataset")
    print(f"Dataset downloaded to directory: {dataset_path_dir}")
    csv_file_name = "PS_20174392719_1491204439457_log.csv"
    csv_file_path = os.path.join(dataset_path_dir, csv_file_name)
    df = pd.read_csv(csv_file_path)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"An error occurred during dataset loading: {e}")
    exit()

df = df.sample(n=50000, random_state=42).reset_index(drop=True)
print(f"\nSampled down to {len(df)} records for this demonstration.")


print("\nConstructing Graph...")
all_customers = pd.concat([df['nameOrig'], df['nameDest']]).unique()
customer_encoder = LabelEncoder()
customer_encoder.fit(all_customers)
num_nodes = len(customer_encoder.classes_)

df['source_node'] = customer_encoder.transform(df['nameOrig'])
df['dest_node'] = customer_encoder.transform(df['nameDest'])

print(f"Graph constructed with {num_nodes} nodes (customers).")


print("\nCreating Node and Edge Features...")
start_time_features = time.time()

all_node_names = customer_encoder.classes_
node_df = pd.DataFrame({'name': all_node_names, 'node_id': range(num_nodes)})

sent_features = df.groupby('nameOrig')['amount'].agg(['sum', 'mean']).rename(
    columns={'sum': 'total_sent', 'mean': 'avg_sent'}
)
received_features = df.groupby('nameDest')['amount'].agg(['sum', 'mean']).rename(
    columns={'sum': 'total_received', 'mean': 'avg_received'}
)

node_df = node_df.merge(sent_features, left_on='name', right_index=True, how='left')
node_df = node_df.merge(received_features, left_on='name', right_index=True, how='left')
node_df['is_merchant'] = node_df['name'].str.startswith('M').astype(int)
node_df = node_df.fillna(0)

print("Generating topological features (Degree, PageRank)...")
G = nx.from_pandas_edgelist(
    df, 'source_node', 'dest_node', create_using=nx.DiGraph()
)
degree_centrality = nx.degree_centrality(G)
pagerank = nx.pagerank(G, alpha=0.85)

node_df['degree_centrality'] = node_df['node_id'].map(degree_centrality).fillna(0)
node_df['pagerank'] = node_df['node_id'].map(pagerank).fillna(0)

node_df = node_df.set_index('node_id').sort_index()
print(f"Node features (including topological) created in {time.time() - start_time_features:.2f} seconds.")


scaler = StandardScaler()
numeric_cols = ['total_sent', 'avg_sent', 'total_received', 'avg_received', 'degree_centrality', 'pagerank']
node_df[numeric_cols] = scaler.fit_transform(node_df[numeric_cols])

X = torch.tensor(node_df.drop(columns=['name']).values, dtype=torch.float)
print(f"Node feature matrix 'X' created with shape: {X.shape}")

edge_index = torch.tensor(df[['source_node', 'dest_node']].values, dtype=torch.long).t().contiguous()

type_dummies = pd.get_dummies(df['type'], prefix='type', dtype=int)
amount_scaler = StandardScaler()
scaled_amount = amount_scaler.fit_transform(df[['amount']])
edge_attr_np = np.concatenate([type_dummies.values, scaled_amount], axis=1)

edge_feature_names = list(type_dummies.columns) + ['amount']
node_feature_names = list(node_df.drop(columns=['name']).columns)

edge_attr = torch.tensor(edge_attr_np, dtype=torch.float)
print(f"Edge attribute matrix 'edge_attr' created with shape: {edge_attr.shape}")

y = torch.tensor(df['isFraud'].values, dtype=torch.long)
print(f"Edge labels 'y' created with shape: {y.shape}")
print("-" * 60)

################################################################################
# Phase 2: Create PyG Data Object and Split
################################################################################
print("Phase 2: Creating PyTorch Geometric Data Object and Splitting...")

data = Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=y)
data.num_nodes = num_nodes

indices = np.arange(data.num_edges)
train_idx, temp_idx = train_test_split(
    indices, test_size=0.4, random_state=42, stratify=data.y.numpy()
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, random_state=42, stratify=data.y[temp_idx].numpy()
)

data.train_mask = torch.zeros(data.num_edges, dtype=torch.bool); data.train_mask[train_idx] = True
data.val_mask = torch.zeros(data.num_edges, dtype=torch.bool); data.val_mask[val_idx] = True
data.test_mask = torch.zeros(data.num_edges, dtype=torch.bool); data.test_mask[test_idx] = True

print(f"Data split into:\n  Training edges: {data.train_mask.sum().item()}\n  Validation edges: {data.val_mask.sum().item()}\n  Test edges: {data.test_mask.sum().item()}")
print("-" * 60)


################################################################################
# Phase 3: GNN Model Definitions
################################################################################
print("Phase 3: Defining the GNN Models (GAT and GraphSAGE)...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
data = data.to(device)

# Model 1: Graph Attention Network (GAT)
class EdgeClassifierGAT(torch.nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, add_self_loops=False)
        self.conv2 = GATConv(hidden_channels, hidden_channels, add_self_loops=False)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_channels + edge_dim, hidden_channels),
            torch.nn.ReLU(), torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_channels, out_channels)
        )
    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        src, dst = edge_index
        combined_features = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        return self.edge_mlp(combined_features)

# Model 2: GraphSAGE
class EdgeClassifierGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_channels + edge_dim, hidden_channels),
            torch.nn.ReLU(), torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_channels, out_channels)
        )
    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        src, dst = edge_index
        combined_features = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        return self.edge_mlp(combined_features)

print("Models defined successfully.")
print("-" * 60)

################################################################################
# Phase 4: Model Training
################################################################################
print("Phase 4: Training GNN Models...")

models = {}
results = {}

for model_name in ["GAT", "GraphSAGE"]:
    print(f"\n--- Training {model_name} Model ---")
    
    if model_name == "GAT":
        model = EdgeClassifierGAT(
            in_channels=data.x.shape[1], edge_dim=data.edge_attr.shape[1],
            hidden_channels=64, out_channels=1
        ).to(device)
    else:
        model = EdgeClassifierGraphSAGE(
            in_channels=data.x.shape[1], edge_dim=data.edge_attr.shape[1],
            hidden_channels=64, out_channels=1
        ).to(device)

    pos_count = data.y[data.train_mask].sum().item()
    weight_value = (data.train_mask.sum().item() / (pos_count + 1e-8)) - 1
    pos_weight = torch.tensor(weight_value, device=device)
    
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    start_time = time.time()
    best_val_auc = 0
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr).squeeze()
        loss = criterion(out[data.train_mask], data.y[data.train_mask].float())
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                out_val = model(data.x, data.edge_index, data.edge_attr).squeeze()
                y_true_val = data.y[data.val_mask].cpu().numpy()
                y_proba_val = torch.sigmoid(out_val[data.val_mask]).cpu().numpy()
                val_auc = roc_auc_score(y_true_val, y_proba_val) if len(np.unique(y_true_val)) > 1 else 0
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val ROC-AUC: {val_auc:.4f}')
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    torch.save(model.state_dict(), f'best_model_{model_name}.pth')

    print(f"✅ {model_name} training complete in {time.time() - start_time:.2f} seconds.")
    models[model_name] = model

print("-" * 60)

################################################################################
# Phase 5: Model Evaluation & Threshold Analysis
################################################################################
print("Phase 5: Final Model Evaluation on Test Set...")
os.makedirs("plots", exist_ok=True)

y_true_test = data.y[data.test_mask].cpu().numpy()

for model_name in ["GAT", "GraphSAGE"]:
    print(f"\n--- Evaluating {model_name} Model ---")
    
    if model_name == "GAT":
        model = EdgeClassifierGAT(in_channels=data.x.shape[1], edge_dim=data.edge_attr.shape[1], hidden_channels=64, out_channels=1).to(device)
    else:
        model = EdgeClassifierGraphSAGE(in_channels=data.x.shape[1], edge_dim=data.edge_attr.shape[1], hidden_channels=64, out_channels=1).to(device)
    
    model.load_state_dict(torch.load(f'best_model_{model_name}.pth'))
    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr).squeeze()
        
    y_proba_test = torch.sigmoid(out[data.test_mask]).cpu().numpy()
    y_pred_test = (y_proba_test > 0.5).astype(int)

    precision_vals, recall_vals, _ = precision_recall_curve(y_true_test, y_proba_test)
    
    results[model_name] = {
        'Accuracy': accuracy_score(y_true_test, y_pred_test),
        'Precision': precision_score(y_true_test, y_pred_test, zero_division=0),
        'Recall': recall_score(y_true_test, y_pred_test, zero_division=0),
        'F1-Score': f1_score(y_true_test, y_pred_test, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true_test, y_proba_test),
        'AUPRC': auc(recall_vals, precision_vals)
    }

    cm = confusion_matrix(y_true_test, y_pred_test)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Not Fraud", "Fraud"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.savefig(f"plots/{model_name}_confusion_matrix.png"); plt.close()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
for model_name in ["GAT", "GraphSAGE"]:
    if model_name == "GAT":
        model = EdgeClassifierGAT(in_channels=data.x.shape[1], edge_dim=data.edge_attr.shape[1], hidden_channels=64, out_channels=1).to(device)
    else:
        model = EdgeClassifierGraphSAGE(in_channels=data.x.shape[1], edge_dim=data.edge_attr.shape[1], hidden_channels=64, out_channels=1).to(device)
    
    model.load_state_dict(torch.load(f'best_model_{model_name}.pth'))
    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr).squeeze()
        y_proba_test = torch.sigmoid(out[data.test_mask]).cpu().numpy()
    
    fpr, tpr, _ = roc_curve(y_true_test, y_proba_test)
    precision_vals, recall_vals, _ = precision_recall_curve(y_true_test, y_proba_test)
    ax1.plot(fpr, tpr, label=f'{model_name} (AUC = {results[model_name]["ROC-AUC"]:.4f})')
    ax2.plot(recall_vals, precision_vals, label=f'{model_name} (AUPRC = {results[model_name]["AUPRC"]:.4f})')

ax1.plot([0, 1], [0, 1], 'k--', label='Random Chance'); ax1.set_title('ROC Curve Comparison'); ax1.set_xlabel('False Positive Rate'); ax1.set_ylabel('True Positive Rate'); ax1.legend()
ax2.set_title('Precision-Recall Curve Comparison'); ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision'); ax2.legend()
plt.savefig("plots/gnn_comparative_curves.png"); plt.close()
print("\nSaved comparative ROC and PR curves.")

results_df = pd.DataFrame(results).T
print("\nModel Performance Summary:")
print(results_df)


best_model_name = results_df['AUPRC'].idxmax()
print(f"\n--- Threshold Optimization for Best Model: {best_model_name} ---")

if best_model_name == "GAT":
    best_model_instance = EdgeClassifierGAT(in_channels=data.x.shape[1], edge_dim=data.edge_attr.shape[1], hidden_channels=64, out_channels=1).to(device)
else:
    best_model_instance = EdgeClassifierGraphSAGE(in_channels=data.x.shape[1], edge_dim=data.edge_attr.shape[1], hidden_channels=64, out_channels=1).to(device)
best_model_instance.load_state_dict(torch.load(f'best_model_{best_model_name}.pth'))
best_model_instance.eval()

with torch.no_grad():
    out = best_model_instance(data.x, data.edge_index, data.edge_attr).squeeze()
    y_proba_test = torch.sigmoid(out[data.test_mask]).cpu().numpy()

### --- FINAL FIX --- ###
precision, recall, thresholds = precision_recall_curve(y_true_test, y_proba_test)

# Calculate F1 score for each threshold. Note precision/recall are 1 element longer than thresholds.
f1_scores = 2 * recall[:-1] * precision[:-1] / (recall[:-1] + precision[:-1] + 1e-8)
best_threshold_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_idx]

plt.figure(figsize=(10, 7))
# Plot precision and recall against their corresponding thresholds
plt.plot(thresholds, precision[:-1], label='Precision', color='blue')
plt.plot(thresholds, recall[:-1], label='Recall', color='green')
### --- END FINAL FIX --- ###

plt.title(f'Precision & Recall vs. Threshold for {best_model_name}')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.axvline(x=best_threshold, color='red', linestyle='--', 
            label=f'Optimal Threshold (F1-Score): {best_threshold:.2f}')
plt.legend()
plt.grid(True)
plt.savefig(f"plots/{best_model_name}_threshold_analysis.png")
plt.close()
print(f"Saved Threshold Analysis plot. Best threshold found at {best_threshold:.4f}")
print("-" * 60)

################################################################################
# Phase 6: Explainable AI (XAI) with CaptumExplainer
################################################################################
print(f"Phase 6: Explaining {best_model_name} Predictions with CaptumExplainer...")
os.makedirs("xai_explanations", exist_ok=True)

test_indices = torch.where(data.test_mask)[0]
fraud_edge_indices_in_test = torch.where(data.y[data.test_mask] == 1)[0]

if len(fraud_edge_indices_in_test) > 0:
    explainer = Explainer(
        model=best_model_instance,
        algorithm=CaptumExplainer('IntegratedGradients'),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='binary_classification', task_level='edge', return_type='probs'
        ),
    )

    first_fraud_edge_original_idx = test_indices[fraud_edge_indices_in_test[0]].item()
    print(f"\nGenerating explanation for a fraudulent transaction (Edge index: {first_fraud_edge_original_idx})...")
    
    explanation = explainer(
        x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
        index=first_fraud_edge_original_idx
    )
    
    node_importance = explanation.x.abs().sum(dim=0).cpu().numpy()
    edge_importance = explanation.edge_attr.abs().sum(dim=0).cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    ax1.barh(node_feature_names, node_importance, color='skyblue'); ax1.set_title('Node Feature Importance')
    ax2.barh(edge_feature_names, edge_importance, color='salmon'); ax2.set_title('Edge Feature Importance')
    plt.tight_layout()
    plt.savefig(f"xai_explanations/{best_model_name}_feature_importance.png"); plt.close()
    print(f"Saved feature importance plot to 'xai_explanations/{best_model_name}_feature_importance.png'")

else:
    print("No fraudulent transactions found in the test set to explain.")
    
print("\n✅ XAI Phase Complete.")
print("=" * 80)
print("Congratulations! Script execution finished successfully.")
print("=" * 80)
