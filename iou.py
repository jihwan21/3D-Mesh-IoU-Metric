import trimesh
import numpy as np

def RightArmIoU(verts_p1, verts_p2, args, vert_idx, smpl_faces):
    # 오른팔에 해당하는 vertices 추출
    vertices_p1 = verts_p1[vert_idx]
    vertices_p2 = verts_p2[vert_idx]

    # 기존의 전체 vertices 인덱스를 오른팔만의 인덱스로 매핑
    # ex) 전체 인덱스 1200 -> 오른팔 인덱스 0
    index_mapping = {index: i for i, index in enumerate(vert_idx)}

    # 오른팔에 해당하는 faces 필터링 및 인덱스 변환
    right_arm_faces = []
    for face in smpl_faces:
        if all(vertex in vert_idx for vertex in face):
            # face의 각 vertex를 새 인덱스로 변환
            new_face = [index_mapping[vertex] for vertex in face]
            right_arm_faces.append(new_face)

    right_arm_faces = np.array(right_arm_faces)

    #####################################
    
    # p1 중점 설정(추후 알고리즘 개발, 임의 설정)
    ### top
    cx1 = (vertices_p1[5][0] + vertices_p1[42][0]) / 2
    cy1 = (vertices_p1[5][1] + vertices_p1[42][1]) / 2
    cz1 = (vertices_p1[5][2] + vertices_p1[42][2]) / 2

    # 중앙 vertices 추가
    test_vertices_p1 = np.vstack((vertices_p1, [cx1, cy1, cz1]))

    # faces 조합 추가
    test_faces_p1 = right_arm_faces

    for k in range(len(args.right_arm_top)):
        if k == len(args.right_arm_top) - 1:
            test_faces_p1 = np.vstack((test_faces_p1, [len(test_vertices_p1)-1, args.right_arm_top[k], args.right_arm_top[0]]))
        else:
            test_faces_p1 = np.vstack((test_faces_p1, [len(test_vertices_p1)-1, args.right_arm_top[k], args.right_arm_top[k+1]]))


    ### down        
    cx2 = (vertices_p1[145][0] + vertices_p1[151][0]) / 2
    cy2 = (vertices_p1[145][1] + vertices_p1[151][1]) / 2
    cz2 = (vertices_p1[145][2] + vertices_p1[151][2]) / 2

    # 중앙 vertices 추가
    test_vertices_p1 = np.vstack((test_vertices_p1, [cx2, cy2, cz2]))  

    for k in range(len(args.right_arm_down)):
        if k == len(args.right_arm_down) - 1:
            test_faces_p1 = np.vstack((test_faces_p1, [len(test_vertices_p1)-1, args.right_arm_down[k], args.right_arm_down[0]]))
        else:
            test_faces_p1 = np.vstack((test_faces_p1, [len(test_vertices_p1)-1, args.right_arm_down[k], args.right_arm_down[k+1]]))
            
    test_p1_mesh = trimesh.Trimesh(vertices=test_vertices_p1, faces=test_faces_p1)
    test_p1_mesh.fix_normals()
    

    # p2 중점 설정(추후 알고리즘 개발, 임의 설정)
    ### top
    cx1 = (vertices_p2[5][0] + vertices_p2[42][0]) / 2
    cy1 = (vertices_p2[5][1] + vertices_p2[42][1]) / 2
    cz1 = (vertices_p2[5][2] + vertices_p2[42][2]) / 2

    # 중앙 vertices 추가
    test_vertices_p2 = np.vstack((vertices_p2, [cx1, cy1, cz1]))

    # faces 조합 추가
    test_faces_p2 = right_arm_faces

    for k in range(len(args.right_arm_top)):
        if k == len(args.right_arm_top) - 1:
            test_faces_p2 = np.vstack((test_faces_p2, [len(test_vertices_p2)-1, args.right_arm_top[k], args.right_arm_top[0]]))
        else:
            test_faces_p2 = np.vstack((test_faces_p2, [len(test_vertices_p2)-1, args.right_arm_top[k], args.right_arm_top[k+1]]))


    ### down        
    cx2 = (vertices_p2[145][0] + vertices_p2[151][0]) / 2
    cy2 = (vertices_p2[145][1] + vertices_p2[151][1]) / 2
    cz2 = (vertices_p2[145][2] + vertices_p2[151][2]) / 2

    # 중앙 vertices 추가
    test_vertices_p2 = np.vstack((test_vertices_p2, [cx2, cy2, cz2]))  

    for k in range(len(args.right_arm_down)):
        if k == len(args.right_arm_down) - 1:
            test_faces_p2 = np.vstack((test_faces_p2, [len(test_vertices_p2)-1, args.right_arm_down[k], args.right_arm_down[0]]))
        else:
            test_faces_p2 = np.vstack((test_faces_p2, [len(test_vertices_p2)-1, args.right_arm_down[k], args.right_arm_down[k+1]]))
        
    test_p2_mesh = trimesh.Trimesh(vertices=test_vertices_p2, faces=test_faces_p2)
    test_p2_mesh.fix_normals()
    
    # 두 메쉬의 교차된 부분 계산
    intersection_mesh = trimesh.boolean.intersection([test_p1_mesh, test_p2_mesh])

    # 교차 부피 계산
    intersection_volume = intersection_mesh.volume

    ### IoU 계산
    # 각 메쉬의 부피 계산
    volume_1 = test_p1_mesh.volume
    volume_2 = test_p2_mesh.volume

    # IoU 계산
    iou = intersection_volume / (volume_1 + volume_2 - intersection_volume)
    iou = iou * 100
    
    return iou