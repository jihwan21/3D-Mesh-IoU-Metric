import numpy as np
import os
import yaml
import json
from easydict import EasyDict as edict

###
def find_closest_points(verts1, verts2):
    """ verts2의 각 점에 대해 verts1에서 가장 가까운 점을 찾음 """
    closest_points = np.zeros_like(verts2)
    for i, point in enumerate(verts2):
        distances = np.linalg.norm(verts1 - point, axis=1)
        closest_points[i] = verts1[np.argmin(distances)]
    return closest_points

def compute_centroid(points):
    """ 포인트들의 중심을 계산 """
    return np.mean(points, axis=0)

def compute_best_fit_transform(A, B):
    """ A와 B 사이의 최적의 변환 행렬을 계산 """
    centroid_A = compute_centroid(A)
    centroid_B = compute_centroid(B)
    
    AA = A - centroid_A
    BB = B - centroid_B
    
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = Vt.T @ U.T
    
    t = centroid_B - R @ centroid_A
    
    return R, t

def normalize_scale(verts):
    """ 포인트 클라우드의 스케일을 정규화 (단위 크기로) """
    scale = np.linalg.norm(verts - compute_centroid(verts), axis=1).max()
    return verts / scale

def icp(A, B, max_iterations=30, tolerance=1e-6):
    """ ICP 알고리즘 수행 """
    prev_error = float('inf')
    A_transformed = A.copy()

    for i in range(max_iterations):
        closest_points = find_closest_points(A_transformed, B)
        
        R, t = compute_best_fit_transform(A_transformed, closest_points)
        
        A_transformed = (R @ A_transformed.T).T + t
        
        mean_error = np.mean(np.linalg.norm(A_transformed - closest_points, axis=1))
        if np.abs(prev_error - mean_error) < tolerance:
            break
        
        prev_error = mean_error

    return A_transformed, R, t, np.abs(prev_error - mean_error) ### 추가

### MotionBERT 코드 
class Loader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream: IO) -> None:
        """Initialise Loader."""

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)

def construct_include(loader: Loader, node: yaml.Node) -> Any:
    """Include file referenced at node."""

    filename = os.path.abspath(os.path.join(loader._root, loader.construct_scalar(node)))
    extension = os.path.splitext(filename)[1].lstrip('.')

    with open(filename, 'r') as f:
        if extension in ('yaml', 'yml'):
            return yaml.load(f, Loader)
        elif extension in ('json', ):
            return json.load(f)
        else:
            return ''.join(f.readlines())
        
def get_config(config_path):
    yaml.add_constructor('!include', construct_include, Loader)
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=Loader)
    config = edict(config)
    _, config_filename = os.path.split(config_path)
    config_name, _ = os.path.splitext(config_filename)
    config.name = config_name
    return config

def get_smpl_vert_config(json_path):
    with open(json_path) as f:
        config = json.load(f)
    return config

### 일대다 매칭 처리 함수
def process_iou_first_last_only(sum_path, iou_values):
    processed_iou = []
    i = 0
    while i < len(sum_path):
        current_frame = sum_path[i][0]
        j = i
        # 같은 첫 번째 frame으로 이루어진 구간 탐색
        while j < len(sum_path) and sum_path[j][0] == current_frame:
            j += 1
        if j - i >= 4:  # 구간 길이가 4 이상인 경우
            processed_iou.append(iou_values[i])  # 맨 처음 IoU 값
            processed_iou.append(iou_values[j - 1])  # 맨 마지막 IoU 값
        else:
            processed_iou.extend(iou_values[i:j])  # 그대로 추가
        i = j
    return processed_iou

def process_iou_average(sum_path, iou_values):
    processed_iou = []
    i = 0
    while i < len(sum_path):
        current_frame = sum_path[i][0]
        j = i
        # 같은 첫 번째 frame으로 이루어진 구간 탐색
        while j < len(sum_path) and sum_path[j][0] == current_frame:
            j += 1
        if j - i >= 4:  # 구간 길이가 4 이상인 경우
            avg_iou = sum(iou_values[i:j]) / (j - i)  # 평균 계산
            processed_iou.append(avg_iou)  # 평균 값을 추가
        else:
            processed_iou.extend(iou_values[i:j])  # 그대로 추가
        i = j
    return processed_iou

