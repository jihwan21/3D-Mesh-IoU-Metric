import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from fastdtw import fastdtw

from .utils import *
from .iou import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/3d_mesh.yaml", help="Path to the config file.")
    parser.add_argument("--smpl_config", type=str, default="config/smpl_vert_segmentation.json", help="Path to the smpl vert index config file.")
    parser.add_argument('--smpl_faces', type=str, default='./npy/smpl_faces.npy', help='Path to the smpl faces file')
    parser.add_argument('-j1', '--joint1_path', type=str, help='joint1 path')
    parser.add_argument('-j2', '--joint2_path', type=str, help='joint2 path')
    parser.add_argument('-v1', '--vert1_path', type=str, help='vert1 path')
    parser.add_argument('-v2', '--vert2_path', type=str, help='vert2 path')
    parser.add_argument('-p', '--match_process', type=str, default='average', help='One-to-many matching processing / first_last or average')
    opts = parser.parse_args()
    return opts

opts = parse_args()
args = get_config(opts.config)
smpl_args = get_smpl_vert_config(opts.smpl_config) # smpl vert index json 파일

# # 두 영상의 예측 joint 좌표 데이터 로드
# joint_p1 = np.load('X3D_p4_golf.npy')
# joint_p2 = np.load('X3D_p5_golf.npy')

# # 두 영상의 메쉬 정점 데이터 로드
# verts_all_p1 = np.load('./npy/verts_all_p4_golf.npy')  # 첫 번째 사람의 메쉬 정점 (6890, 3)
# verts_all_p2 = np.load('./npy/verts_all_p5_golf.npy')  # 두 번째 사람의 메쉬 정점 (6890, 3)
# smpl_faces = np.load('./npy/smpl_faces.npy')

# 두 영상의 예측 joint 좌표 데이터 로드
joint_p1 = np.load(opts.joint1_path)
joint_p2 = np.load(opts.joint2_path)

# 두 영상의 메쉬 정점 데이터 로드
verts_all_p1 = np.load(opts.vert1_path)
verts_all_p2 = np.load(opts.vert2_path)

smpl_faces = np.load(opts.smpl_faces)

# # 오른 손목 y 좌표만을 대상으로 DTW
# joint_p4_y = {}
# joint_p5_y = {}

# for i in range(len(joint_p4)):
#     y = joint_p4[i][16][1] # 오른손목 y 좌표
#     joint_p4_y[i] = y 

# for i in range(len(joint_p5)):
#     y = joint_p5[i][16][1] # 오른손목 y 좌표
#     joint_p5_y[i] = y 

# distance, path = fastdtw(list(joint_p4_y.values()), list(joint_p5_y.values()), dist=2)
# path = [(x, y) for x, y in path]


# SwingNet 결과로 Key Frmae 선정 후 DTW 매칭
key1_p4 = joint_p1[:73] 
key1_p5 = joint_p2[:97] 

key2_p4 = joint_p1[73:112] 
key2_p5 = joint_p2[97:137] 

key3_p4 = joint_p1[112:124] 
key3_p5 = joint_p2[137:148] 

key4_p4 = joint_p1[124:131] 
key4_p5 = joint_p2[148:153]

key5_p4 = joint_p1[131:] 
key5_p5 = joint_p2[153:]

distance1, path = fastdtw(key1_p4, key1_p5, dist=2)
path_1 = [(x, y) for x, y in path]

distance, path = fastdtw(key2_p4, key2_p5, dist=2)
path_2 = [(x, y) for x, y in path]

distance, path = fastdtw(key3_p4, key3_p5, dist=2)
path_3 = [(x, y) for x, y in path]

distance, path = fastdtw(key4_p4, key4_p5, dist=2)
path_4 = [(x, y) for x, y in path]

distance, path = fastdtw(key5_p4, key5_p5, dist=2)
path_5 = [(x, y) for x, y in path]

sum_path = path_1 + path_2 + path_3 + path_4 + path_5



# 전체 frame pair mean error 기록하여 기준 회전행렬, 이동벡터 정하는 과정(생략)
# 0번째 프레임의 회전행렬, 이동벡터 값 활용
verts_p1_normalized = normalize_scale(verts_all_p1[0])
verts_p2_normalized = normalize_scale(verts_all_p2[0])

_, rotation, translation, _ = icp(verts_p1_normalized, verts_p2_normalized)

# 기준 회전행렬, 이동벡터 변수 지정
R = rotation
t = translation


# 부위별 IoU 계산
arm_IoU = []
fore_arm_IoU = []
hand_IoU = []

for idx_1, idx_2 in sum_path:
    # 스케일 정규화
    verts_p1_normalized = normalize_scale(verts_all_p1[idx_1])
    verts_p2_normalized = normalize_scale(verts_all_p2[idx_2])

    # ICP 실행 X / 기준 R, t 적용
    aligned_verts_p2 = (R @ verts_p2_normalized.T).T + t
    
    right_arm_iou = RightArmIoU(verts_p1_normalized, aligned_verts_p2, args, smpl_args['rightArm'], smpl_faces)
    arm_IoU.append(right_arm_iou)
    
    right_forearm_iou = RightForeArmIoU()
    fore_arm_IoU.append(right_forearm_iou)

    hand_iou = RightHandIoU()
    hand_IoU.append(hand_iou)

    # 일대다 매칭 처리
    if opts.match_process == 'average':
        process_arm_IoU = process_iou_average(sum_path, arm_IoU)
    else:
        process_arm_IoU = process_iou_first_last_only(sum_path, arm_IoU)