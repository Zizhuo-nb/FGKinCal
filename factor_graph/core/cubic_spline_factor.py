'''
input :  two point clouds from preprocessing
output : the residual vector
'''
import numpy as np
from src.config.sICPconfig import sICPconfig
from sklearn.neighbors import NearestNeighbors
from factor_graph.tool.tool import c0_between_factor_cubic_spline_jacobian,c1_between_factor_cubic_spline_jacobian,c2_between_factor_cubic_spline_jacobian  
from factor_graph.tool.tool import icp_error_cubic_spline_jacobian
from factor_graph.tool.tool import transform_points_with_cubic_spline
import random

class icpFactor:
    def __init__(self, pc1, pc2):
        self.pc1 = pc1
        self.pc2 = pc2
        self.pc_i = None # point matches 1 & 2 + averaged normal vector
        self.xyzm1 = None # point matches 1
        self.xyzm2 = None  # point matches 2
        
    
    
    def matching(self, config : sICPconfig,time2):
        time2 = np.asarray(time2).reshape(-1)
        pc1_downsampled, _ = self.voxel_downsampling(self.pc1, config.voxel_size)

        nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(self.pc2)
        dNN, idxNN = nbrs.kneighbors(pc1_downsampled)

        valid_mask = dNN[:,0] < config.max_dist
        right_idx = idxNN[valid_mask, 0]
        self.mx1 = pc1_downsampled[valid_mask]
        self.mx2 = self.pc2[right_idx]
        matched_time = time2[right_idx]
        self.pc_i = np.hstack((self.mx1, self.mx2))

        n1,std1,idx = icpFactor.normals(self.mx1, self.pc1, config.normals_radius, config.normals_minpoints, config.normals_maxpoints)
        self.mx1, self.mx2 = self.mx1[idx], self.mx2[idx]
        matched_time = matched_time[idx]

        # Filter by roughness
        if config.roughness_filter_use:
            idx = std1 <= config.max_roughness
            self.mx1, self.mx2, n1 = self.mx1[idx], self.mx2[idx], n1[idx]
            matched_time = matched_time[idx]

        # Compute normals and roughness for pc2
        
        n2, std2, idx = icpFactor.normals(self.mx2, self.pc2, config.normals_radius, config.normals_minpoints, config.normals_maxpoints) #n: [nx, ny, nz, roughness]
        self.mx1, self.mx2, n1 = self.mx1 [idx], self.mx2[idx], n1[idx]
        matched_time = matched_time[idx]

        # Filter by roughness value 
        if config.roughness_filter_use:
            idx = std2 <= config.max_roughness
            self.mx1 , self.mx2, n1, n2 = self.mx1 [idx], self.mx2[idx], n1[idx], n2[idx]
            matched_time = matched_time[idx]

        # Compute sum of scalar product

        # 保证只有一个法向量时仍然是二维数组 (1, 3)
        n1 = np.asarray(n1).reshape(-1, 3)
        n2 = np.asarray(n2).reshape(-1, 3)

        # 所有匹配都被过滤掉
        if n1.shape[0] == 0 or n2.shape[0] == 0:
            return None, None

        sp = np.sum(n1 * n2, axis=1)

        # Filter by angle between normals
        if config.normal_angle_use:
            alpha_max = np.radians(config.normal_angle_max) 
            th = np.cos(alpha_max)
            idx = np.abs(sp) >= th
            self.mx1, self.mx2, n1, n2, sp = self.mx1 [idx], self.mx2[idx], n1[idx], n2[idx], sp[idx]
            matched_time = matched_time[idx]

        # Compute mean normal and point-to-plane distance
        idx = sp < 0
        n2[idx] = -n2[idx]
        n = 0.5 * (n1 + n2)
        dx = self.mx2 - self.mx1 
        p2p_d = np.sum(n * dx, axis=1)
       
        # Filter by point-to-plane MAD
        if config.mad_use:
            s_mad = 1.4826 * np.median(np.abs(p2p_d - np.median(p2p_d)))
            idx = np.abs(p2p_d - np.median(p2p_d)) <= 3 * s_mad
            self.mx1, self.mx2, n = self.mx1 [idx], self.mx2[idx], n[idx]
            matched_time = matched_time[idx]

        # Update filtered matches   最终 self.pc_i 的每一行是：[x1, y1, z1, x2, y2, z2, nx, ny, nz],这个会进入进行计算
        self.pc_i = np.hstack((self.mx1 , self.mx2, n))

        # Matching points
        self.xyzm1 = self.mx1
        self.xyzm2 = self.mx2

        return self.pc_i,matched_time
    
    
    
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

                plane_normal, std_dev = icpFactor.plane_fitting(neighbors)
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
    
    


    
def icp_residual_for_spline(p1, p2, n, coefficients, timeL):
    p2_new = transform_points_with_cubic_spline(
        p2,
        coefficients,
        timeL
    )

    residual = np.sum(n * (p2_new - p1), axis=1)

    return residual, p2_new



def icp_error_func_cubic_spline(p1, p2, n, timeL):

    def error_func(this, values, H):
        key = this.keys()[0]

        # 读取24维样条系数，不再读取Pose3
        coefficients = values.atVector(key)

        residual, p2_new = icp_residual_for_spline(
            p1,
            p2,
            n,
            coefficients,
            timeL
        )

        if H is not None:
            # 左乘雅可比必须使用当前修正后的点 p2_new
            H[0] = icp_error_cubic_spline_jacobian(
                p2_new,
                n,
                timeL
            )

        return residual

    return error_func









#=============================between factor for cubic spline=========================

def c0_between_error_func(
    base_coeff_left,
    base_coeff_right,
    timeL
):
    """
    base_coeff_left/right:
        两个窗口当前已有的完整24维样条系数

    timeL:
        左侧样条段的末端局部时间

    residual:
        spline_left(timeL) - spline_right(0)
        维度为6
    """

    base_coeff_left = np.asarray(base_coeff_left).reshape(4, 6)
    base_coeff_right = np.asarray(base_coeff_right).reshape(4, 6)

    def error_func(this, values, H):
        key_left = this.keys()[0]
        key_right = this.keys()[1]

        # GTSAM本轮优化的新增量
        delta_left = values.atVector(key_left).reshape(4, 6)
        delta_right = values.atVector(key_right).reshape(4, 6)

        # 更新后的完整样条系数
        coeff_left = base_coeff_left + delta_left
        coeff_right = base_coeff_right + delta_right

        # 左段末端
        value_left = (
            coeff_left[0]
            + timeL * coeff_left[1]
            + timeL**2 * coeff_left[2]
            + timeL**3 * coeff_left[3]
        )

        # 右段起点 t=0
        value_right = coeff_right[0]

        residual = value_left - value_right

        if H is not None:
            H_left, H_right = (
                c0_between_factor_cubic_spline_jacobian(timeL)
            )

            H[0] = H_left
            H[1] = H_right

        return residual

    return error_func


def c1_between_error_func(
    base_coeff_left,
    base_coeff_right,
    timeL
):
    """
    C1连续：
    spline_left'(timeL) - spline_right'(0)
    """

    base_coeff_left = np.asarray(
        base_coeff_left
    ).reshape(4, 6)

    base_coeff_right = np.asarray(
        base_coeff_right
    ).reshape(4, 6)

    def error_func(this, values, H):
        key_left = this.keys()[0]
        key_right = this.keys()[1]

        # GTSAM本轮优化的新增量
        delta_left = values.atVector(
            key_left
        ).reshape(4, 6)

        delta_right = values.atVector(
            key_right
        ).reshape(4, 6)

        # 更新后的完整样条系数
        coeff_left = base_coeff_left + delta_left
        coeff_right = base_coeff_right + delta_right

        # 左段在timeL处的一阶导数
        derivative_left = (
            coeff_left[1]
            + 2 * timeL * coeff_left[2]
            + 3 * timeL**2 * coeff_left[3]
        )

        # 右段在起点t=0处的一阶导数
        derivative_right = coeff_right[1]

        residual = derivative_left - derivative_right

        if H is not None:
            H_left, H_right = (
                c1_between_factor_cubic_spline_jacobian(
                    timeL
                )
            )

            H[0] = H_left
            H[1] = H_right

        return residual

    return error_func



def c2_between_error_func(
    base_coeff_left,
    base_coeff_right,
    timeL
):
    """
    C2连续：
    spline_left''(timeL) - spline_right''(0)
    """

    base_coeff_left = np.asarray(
        base_coeff_left
    ).reshape(4, 6)

    base_coeff_right = np.asarray(
        base_coeff_right
    ).reshape(4, 6)

    def error_func(this, values, H):
        key_left = this.keys()[0]
        key_right = this.keys()[1]

        # GTSAM本轮优化的新增量
        delta_left = values.atVector(
            key_left
        ).reshape(4, 6)

        delta_right = values.atVector(
            key_right
        ).reshape(4, 6)

        # 更新后的完整样条系数
        coeff_left = base_coeff_left + delta_left
        coeff_right = base_coeff_right + delta_right

        # 左段在timeL处的二阶导数
        second_derivative_left = (
            2 * coeff_left[2]
            + 6 * timeL * coeff_left[3]
        )

        # 右段在起点t=0处的二阶导数
        second_derivative_right = (
            2 * coeff_right[2]
        )

        residual = (
            second_derivative_left
            - second_derivative_right
        )

        if H is not None:
            H_left, H_right = (
                c2_between_factor_cubic_spline_jacobian(
                    timeL
                )
            )

            H[0] = H_left
            H[1] = H_right

        return residual

    return error_func