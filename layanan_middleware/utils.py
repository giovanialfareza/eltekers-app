# utils.py - Berisi semua kelas dan fungsi pendukung dari kode Anda

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from ultralytics import YOLO

# --- Konstanta Global dari kode Anda ---
NUM_NODES = 17
IN_CHANNELS = 2  # x, y coordinates
SEQ_LENGTH = 30
# NUM_CLASSES akan diatur saat model dimuat

# --- Class GraphConvolution dari kode Anda ---
class GraphConvolution(nn.Module):
  def __init__(self, in_features, out_features):
    super(GraphConvolution, self).__init__()
    self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
    self.in_features = in_features
    self.out_features = out_features
    self.reset_parameters()

  def reset_parameters(self):
    nn.init.kaiming_uniform_(self.weight)

  def forward(self, input, adj):
    support = torch.einsum("biv,io->bov", input, self.weight)
    output = torch.einsum("uv,bov->bou", adj, support)
    return output

# --- Class STGCN dari kode Anda ---
class STGCN(nn.Module):
  def __init__(self, in_channels, num_nodes, num_classes, seq_length, A):
    super(STGCN, self).__init__()
    self.num_nodes = num_nodes
    self.seq_length = seq_length
    self.register_buffer('A_tensor', torch.tensor(A, dtype=torch.float32, requires_grad=False))

    # Spatial Temporal Graph Convolutional Block 1
    self.st_gcn1 = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=(1,1), padding=(0,0)),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, num_nodes)),
        nn.ReLU()
    )
    self.gcn1 = GraphConvolution(64, 64)

    # Spatial Temporal Graph Convolutional Block 2
    self.st_gcn2 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=(1,1), padding=(0,0)),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, num_nodes)),
        nn.ReLU()
    )
    self.gcn2 = GraphConvolution(128, 128)

    # Classification Head
    self.fc = nn.Linear(128 * num_nodes, num_classes)

  def forward(self, x):
    # First Block
    A = self.A_tensor
    x = self.st_gcn1(x)
    x = self.gcn1(x.squeeze(2), A).unsqueeze(2)
    # Second Block
    x = self.st_gcn2(x)
    x = self.gcn2(x.squeeze(2), A).unsqueeze(2)

    # Flatten for classification
    x = x.reshape(x.size(0), -1)
    logits = self.fc(x)
    return logits

# --- Class InferenceDataPreprocessor dari kode Anda ---
class InferenceDataPreprocessor:
  def __init__(self, yolo_model_path='yolov8n-pose.pt', device='auto'):
    if device == 'auto':
      self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
      self.device = device
    self.model = YOLO(yolo_model_path).to(self.device)

  def normalize_keypoints(self, keypoints, frame_width, frame_height):
    normalized_kpts = keypoints.copy()
    normalized_kpts[:, 0] = normalized_kpts[:, 0] / frame_width
    normalized_kpts[:, 1] = normalized_kpts[:, 1] / frame_height
    return normalized_kpts

  def extract_keypoints(self, frame):
    results = self.model(frame, verbose=False, device=self.device)
    best_person_keypoints_normalized = None
    best_person_keypoints_raw = None
    max_area = 0

    if not results or not results[0].boxes:
      return None, None

    h, w, _ = frame.shape

    for r in results:
      if r.keypoints and r.keypoints.xy.shape[0] > 0 and r.keypoints.xy.shape[1] == 17 \
         and r.boxes and r.boxes.xyxy.shape[0] > 0:
        bbox = r.boxes.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = bbox
        current_area = (x2 - x1) * (y2 - y1)

        if current_area > max_area:
          max_area = current_area
          keypoints_raw = r.keypoints.xy[0].cpu().numpy()
          if np.sum(np.isnan(keypoints_raw)) == 0 and np.sum(keypoints_raw) != 0:
            best_person_keypoints_raw = keypoints_raw
            best_person_keypoints_normalized = self.normalize_keypoints(keypoints_raw, w, h)
          else:
            best_person_keypoints_raw = None
            best_person_keypoints_normalized = None
            max_area = 0
    return best_person_keypoints_normalized, best_person_keypoints_raw

# --- Fungsi seq_to_tensor dari kode Anda ---
def seq_to_tensor(seq_np):
    # Mengubah sekuens numpy array ke tensor yang siap untuk model
    return torch.FloatTensor(seq_np).permute(2, 0, 1)

# --- Fungsi build_adjacency_matrix dari kode Anda ---
def build_adjacency_matrix(num_nodes=17):
    # Menggunakan representasi graf pose dari kode Anda
    adj = np.eye(num_nodes)
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Wajah
        (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # Lengan
        (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),  # Kaki
        (5, 11), (6, 12),  # Torso
        (0, 5), (0, 6)  # Bahu ke hidung
    ]
    for i, j in connections:
        adj[i, j] = 1
        adj[j, i] = 1

    D = np.sum(adj, axis=1)
    D_inv_sqrt = np.power(D, -0.5).flatten()
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
    D_inv_sqrt = np.diag(D_inv_sqrt)
    A_norm = adj.dot(D_inv_sqrt).transpose().dot(D_inv_sqrt)
    return A_norm

# --- Fungsi draw_keypoints_and_connections dari kode Anda ---
def draw_keypoints_and_connections(frame, keypoints, predicted_label="Unknown"):
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Wajah
        (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # Lengan
        (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),  # Kaki
        (5, 11), (6, 12),  # Torso
        (0, 5), (0, 6)  # Bahu ke hidung
    ]

    for i, j in connections:
        if keypoints[i, 0] != 0 and keypoints[i, 1] != 0 and \
           keypoints[j, 0] != 0 and keypoints[j, 1] != 0:
            p1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
            p2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
            cv2.line(frame, p1, p2, (0, 255, 0), 2)  # Green lines

    for i in range(keypoints.shape[0]):
        x, y = int(keypoints[i, 0]), int(keypoints[i, 1])
        if x != 0 and y != 0:
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Red circles

    cv2.putText(frame, predicted_label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    return frame
