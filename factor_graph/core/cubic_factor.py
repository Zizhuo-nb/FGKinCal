import numpy as np
from src.config.sICPconfig import sICPconfig
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation
from factor_graph.tool.tool import rigid_transform
from factor_graph.tool.tool import icp_error_jacobian
import random


class CubicIcpFactor:
    def __init__(self,pcl,pcr):
        self.pc1 = pcl
        self.pc2 = pcr
        self.pc_i = None # point matches 1 & 2 + averaged normal vector
        self.xyzm1 = None # point matches 1
        self.xyzm2 = None  # point matches 2

    def matching(self, config : sICPconfig):
            pc1_downsampled, _ = self.voxel_downsampling(self.pc1, config.voxel_size)
    
            nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(self.pc2)
            dNN, idxNN = nbrs.kneighbors(pc1_downsampled)
    
            valid_mask = dNN[:,0] < config.max_dist
    
            self.mx1 = pc1_downsampled[valid_mask]

            self.idx2 = idxNN[valid_mask, 0]   
            self.mx2 = self.pc2[self.idx2]
            self.pc_i = np.hstack((self.mx1, self.mx2))
    
            n1, std1, idx = CubicIcpFactor.normals(
                self.mx1,
                self.pc1,
                config.normals_radius,
                config.normals_minpoints,
                config.normals_maxpoints
            )

            self.mx1 = self.mx1[idx]
            self.mx2 = self.mx2[idx]
            self.idx2 = self.idx2[idx]
    
            # Filter by roughness
            if config.roughness_filter_use:
                idx = std1 <= config.max_roughness

                self.mx1 = self.mx1[idx]
                self.mx2 = self.mx2[idx]
                self.idx2 = self.idx2[idx]
                n1 = n1[idx]
    
            # Compute normals and roughness for pc2
            
            n2, std2, idx = CubicIcpFactor.normals(
                self.mx2,
                self.pc2,
                config.normals_radius,
                config.normals_minpoints,
                config.normals_maxpoints
            )

            self.mx1 = self.mx1[idx]
            self.mx2 = self.mx2[idx]
            self.idx2 = self.idx2[idx]
            n1 = n1[idx]
    
            # Filter by roughness value 
            if config.roughness_filter_use:
                idx = std2 <= config.max_roughness

                self.mx1 = self.mx1[idx]
                self.mx2 = self.mx2[idx]
                self.idx2 = self.idx2[idx]
                n1 = n1[idx]
                n2 = n2[idx]
    
            # Compute sum of scalar product
            sp = np.sum(n1 * n2, axis=1)
    
            # Filter by angle between normals
            if config.normal_angle_use:
                alpha_max = np.radians(config.normal_angle_max)
                th = np.cos(alpha_max)

                idx = np.abs(sp) >= th

                self.mx1 = self.mx1[idx]
                self.mx2 = self.mx2[idx]
                self.idx2 = self.idx2[idx]
                n1 = n1[idx]
                n2 = n2[idx]
                sp = sp[idx]
    
            # Compute mean normal and point-to-plane distance
            idx = sp < 0
            n2[idx] = -n2[idx]
            n = 0.5 * (n1 + n2)
            dx = self.mx2 - self.mx1 
            p2p_d = np.sum(n * dx, axis=1)
           
            # Filter by point-to-plane MAD
            if config.mad_use:
                median = np.median(p2p_d)
                s_mad = 1.4826 * np.median(np.abs(p2p_d - median))

                idx = np.abs(p2p_d - median) <= 3 * s_mad

                self.mx1 = self.mx1[idx]
                self.mx2 = self.mx2[idx]
                self.idx2 = self.idx2[idx]
                n = n[idx]
    
            # Update filtered matches   最终 self.pc_i 的每一行是：[x1, y1, z1, x2, y2, z2, nx, ny, nz],这个会进入进行计算
            self.pc_i = np.hstack((self.mx1, self.mx2, n))

            self.xyzm1 = self.mx1
            self.xyzm2 = self.mx2

            return self.pc_i, self.idx2
    
        
        
        
    @staticmethod
    def voxel_downsampling(points, voxel_size):
        
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)
        _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
        return points[unique_indices], unique_indices


    @staticmethod
    def normals(x,pc,r,minPts,maxPts):

        """ Computes the surface normal vectors for input point cloud

        Args:
            pc: point cloud as numpy array [Nx3]
            r: neighborhood point radius
            minPts: minimum points threshold to estimate a valid normal vector
        
        Returns:
            n: Normal vectors as numpy array
            std: roughness value of the local plane fit
            idx: indices of valid normal esimations
        """
        nbrs = NearestNeighbors(radius=r, algorithm='auto').fit(pc)
        n = []
        std = []

        discarded_indices = []
        for idx, point in enumerate(x):
            _, indices = nbrs.radius_neighbors([point])
            if len(indices[0]) >= minPts:
                neighbors = pc[indices[0]]

                # random subsampling of the neighbors if the number is too large
                if len(neighbors) > maxPts: # TODO: read from config file !  
                    idx_r = random.sample(range(0, len(neighbors)-1), maxPts)
                    neighbors = neighbors[idx_r]

                plane_normal, std_dev = CubicIcpFactor.plane_fitting(neighbors)
                n.append(plane_normal)
                std.append(std_dev)
            else:
                discarded_indices.append(idx)
        kept_indices = [i for i in range(len(x)) if i not in discarded_indices]
        return np.array(n), np.array(std), kept_indices
    
    @staticmethod
    def plane_fitting(points):
        """ Computes the plane parameters for input points

        Args:
            points: 3D points as numps array

        Returns:
            plane_normal: Normal vectors as numpy array
            std_dev: roughness value of the local plane fit
        """
        # compute normal
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        _, _, vh = np.linalg.svd(centered_points)
        plane_normal = vh[-1, :]
        # compute point2plane distances
        d = -np.dot(plane_normal, centroid)
        distances = np.abs(np.dot(points, plane_normal) + d) / np.linalg.norm(plane_normal)
        # compute rougness
        variance_factor = np.sum(distances**2) / (points.shape[0] - 3)
        std_dev = np.sqrt(variance_factor)
        return plane_normal, std_dev   # [nx,ny,nz,roughness]






def icp_error_func(
        coefficients,
        matching,
        pcr_idx,
        time_r,
        R_NB_r,
        t_NB_r,
        window_start,
        window_duration
):
    """
coefficients: (24,)
    order:
    [rx_a0, rx_a1, rx_a2, rx_a3,
        ry_a0, ry_a1, ry_a2, ry_a3,
        rz_a0, rz_a1, rz_a2, rz_a3,
        tx_a0, tx_a1, tx_a2, tx_a3,
        ty_a0, ty_a1, ty_a2, ty_a3,
        tz_a0, tz_a1, tz_a2, tz_a3]

matching: (N, 9)
    [left_UTM, right_static_UTM, normal_UTM]

return:
    residual: (N,)
    jacobian: (N, 24)
"""
    p_left = matching[:,0:3]
    p_right_static = matching[:,3:6]
    normals = matching[:,6:9]
    time_right = time_r[pcr_idx]
    R_NB_right = R_NB_r[pcr_idx]
    t_NB_right = t_NB_r[pcr_idx]

    #right UTM point back to body frame
    q_static_body = np.einsum(
        "nij,nj->ni",
        R_NB_right.transpose(0,2,1),
        p_right_static - t_NB_right)

    #spline base function
    u = (time_right-window_start)/window_duration

    basis = np.column_stack((
        np.ones_like(u),
        u,
        u**2,
        u**3
    )) #(N,4)

    coefficients = np.asarray(coefficients).reshape(6,4)

    #6dof kin calibration
    xi = basis @ coefficients.T #(N,6)
    rotation_vector = xi[:,0:3]
    translation = xi[:,3:6]
    #kin cal fix static point
    delta_R = Rotation.from_rotvec(rotation_vector).as_matrix()

    q_corrected_body = (
        np.einsum("nij,nj->ni", delta_R, q_static_body)
        + translation
    )
    #back to UTM
    p_right_corrected = (
        np.einsum("nij,nj->ni", R_NB_right, q_corrected_body)
        + t_NB_right
    )
    #point_to_plane error
    residual = np.sum(
        normals * (p_right_corrected - p_left),
        axis=1
    )  # (N,)
    # ============================================================
    # 9. 对瞬时6D动态外参的雅可比
    #
    # h_i =
    # n_i^T R_NB(t_i)
    # [-[q_corrected^B]_x, I]
    #
    # 首先计算：
    # g_i^T = n_i^T R_NB(t_i)
    # ============================================================

    g = np.einsum(
        "ni,nij->nj",
        normals,
        R_NB_right
    )  # (N, 3)

    # g^T(-[q]_x) = q × g
    jacobian_rotation = np.cross(
        q_corrected_body,
        g
    )  # (N, 3)

    jacobian_translation = g  # (N, 3)

    jacobian_pose = np.hstack((
        jacobian_rotation,
        jacobian_translation
    ))  # (N, 6)

    # ============================================================
    # 10. 由6维位姿雅可比扩展为24维样条系数雅可比
    #
    # J_i = h_i ⊗ b(u_i)^T
    #
    # 每个自由度对应4个样条系数
    # ============================================================

    jacobian = (
        jacobian_pose[:, :, None]
        * basis[:, None, :]
    ).reshape(-1, 24)

    return residual, jacobian

    

    