# pems_bay_dataset_config.py

import os
import h5py
import pickle
import numpy as np
import pandas as pd
import networkx as nx

# 1) 데이터 로드
BASE_DIR = os.path.join('pems_bay_dataset')
# — 시계열 데이터(.h5)에서 축 정보만 쓰지만, 노드는 메타에서 로드
META_H5 = os.path.join(BASE_DIR, 'pems-bay-meta.h5')
# — adj matrix
ADJ_PKL = os.path.join(BASE_DIR, 'adj_mx_bay.pkl')

# 2) 인접 행렬 불러오기
with open(ADJ_PKL, 'rb') as f:
    adj_data = pickle.load(f, encoding='latin1')

# 리스트인 경우, 3번째 요소가 (N,N) 행렬인지 확인
if isinstance(adj_data, list):
    # 만약 adj_data[2]가 numpy.ndarray이고 정사각행렬이면 그걸 채택
    arr2 = adj_data[2]
    if isinstance(arr2, np.ndarray) and arr2.ndim == 2 and arr2.shape[0]==arr2.shape[1]:
        adj = arr2
    else:
        raise ValueError("adj_mx_bay.pkl에서 올바른 행렬을 찾을 수 없습니다.")
elif isinstance(adj_data, dict):
    # dict 안에 'adj_mx' 키가 있는 경우
    adj = adj_data.get('adj_mx') or adj_data.get('adj_mx_bay')
else:
    adj = np.array(adj_data)

node_adj_mat = adj.astype(float)


# 3) 메타데이터에서 노드 좌표 가져오기
with h5py.File(META_H5, 'r') as f:
    meta_grp = f['meta']
    # 블록0에는 ['City','Abs_PM','Latitude','Longitude','Length']
    # 블록1에는 ['Fwy','District','County','Lanes','User_ID_4']
    items0 = meta_grp['block0_items'][:].astype(str)
    vals0  = meta_grp['block0_values'][:]
    items1 = meta_grp['block1_items'][:].astype(str)
    vals1  = meta_grp['block1_values'][:]
    cols = np.concatenate([items0, items1])
    vals = np.concatenate([vals0, vals1], axis=1)
    meta_df = pd.DataFrame(vals, columns=cols)

# 예: Latitude, Longitude를 coords로 사용
coords = meta_df[['Latitude','Longitude']].values
num_nodes = node_adj_mat.shape[0]

# 4) 노드 리스트 생성
#    각 노드: dict(id=int, coords=(lat,lon))
converted_nodes = [
    {'id': int(i), 'coords': tuple(coords[i])}
    for i in range(num_nodes)
]

# 5) 간선 리스트 생성
#    인접 행렬의 비영(>0) 위치에 대해 양방향 간선 생성
converted_edges = []
for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j and node_adj_mat[i, j] > 0:
            converted_edges.append({
                'start': int(i),
                'end':   int(j),
                'distance': float(node_adj_mat[i, j])
            })

# 6) edge_index, edge_attr 정의
#    PyG 스타일: edge_index shape=(2, E), edge_attr shape=(E, 1)
edge_index = np.array(
    [[e['start'] for e in converted_edges],
     [e['end']   for e in converted_edges]],
    dtype=int
)
edge_attr = np.array(
    [[e['distance']] for e in converted_edges],
    dtype=float
)

# 7) 노드 차수 리스트 (degree)
node_degree_list = node_adj_mat.sum(axis=1).tolist()

# 8) 노드 간 최단경로 거리 (all-pairs shortest path)
#    networkx를 사용해 가중치 그래프 생성 후 계산
G = nx.from_numpy_array(node_adj_mat, create_using=nx.DiGraph())
# 가중치는 'weight' 속성명에 저장됨
for u, v, d in G.edges(data=True):
    d['weight'] = node_adj_mat[u, v]
# 모든 쌍 최단거리
spd_dict = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
# matrix 형태로 변환
node_spd = np.zeros((num_nodes, num_nodes), dtype=float)
for i in range(num_nodes):
    lengths = spd_dict[i]
    for j in range(num_nodes):
        node_spd[i, j] = lengths.get(j, np.inf)

# --- 모듈 인터페이스 ---
# converted_nodes:   [{'id':..., 'coords':(...,...)}, …]
# converted_edges:   [{'start':…, 'end':…, 'distance':…}, …]
# node_adj_mat:      np.ndarray, shape=(N,N)
# node_degree_list:  list of length N
# node_spd:          np.ndarray, shape=(N,N)
# edge_index:        np.ndarray, shape=(2, E)
# edge_attr:         np.ndarray, shape=(E, 1)
