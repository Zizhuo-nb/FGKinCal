'''
input :  two point clouds from preprocessing
output : the residual vector
'''
import numpy as np
from src.config.sICPconfig import sICPconfig
from sklearn.neighbors import NearestNeighbors
from factor_graph.tool.tool import rigid_transform
from factor_graph.tool.tool import icp_error_jacobian
import random

class icpFactorbefore:
    def __init__(self, pc1, pc2):
        self.pc1 = pc1
        self.pc2 = pc2
        self.pc_i = None # point matches 1 & 2 + averaged normal vector
        self.xyzm1 = None # point matches 1
        self.xyzm2 = None  # point matches 2
        
    
    
    def matching(self, config : sICPconfig):
        pc1_downsampled, _ = self.voxel_downsampling(self.pc1, config.voxel_size)

        nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(self.pc2)
        dNN, idxNN = nbrs.kneighbors(pc1_downsampled)

        valid_mask = dNN[:,0] < config.max_dist

        self.mx1 = pc1_downsampled[valid_mask]
        self.mx2 = self.pc2[idxNN[valid_mask, 0]]
        self.pc_i = np.hstack((self.mx1, self.mx2))

        n1,std1,idx = icpFactor.normals(self.mx1, self.pc1, config.normals_radius, config.normals_minpoints, config.normals_maxpoints)
        self.mx1, self.mx2 = self.mx1[idx], self.mx2[idx]

        # Filter by roughness
        if config.roughness_filter_use:
            idx = std1 <= config.max_roughness
            self.mx1, self.mx2, n1 = self.mx1[idx], self.mx2[idx], n1[idx]

        # Compute normals and roughness for pc2
        
        n2, std2, idx = icpFactor.normals(self.mx2, self.pc2, config.normals_radius, config.normals_minpoints, config.normals_maxpoints) #n: [nx, ny, nz, roughness]
        self.mx1, self.mx2, n1 = self.mx1 [idx], self.mx2[idx], n1[idx]

        # Filter by roughness value 
        if config.roughness_filter_use:
            idx = std2 <= config.max_roughness
            self.mx1 , self.mx2, n1, n2 = self.mx1 [idx], self.mx2[idx], n1[idx], n2[idx]

        # Compute sum of scalar product
        sp = np.sum(n1 * n2, axis=1)

        # Filter by angle between normals
        if config.normal_angle_use:
            alpha_max = np.radians(config.normal_angle_max) 
            th = np.cos(alpha_max)
            idx = np.abs(sp) >= th
            self.mx1, self.mx2, n1, n2, sp = self.mx1 [idx], self.mx2[idx], n1[idx], n2[idx], sp[idx]

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

        # Update filtered matches   最终 self.pc_i 的每一行是：[x1, y1, z1, x2, y2, z2, nx, ny, nz],这个会进入进行计算
        self.pc_i = np.hstack((self.mx1 , self.mx2, n))

        # Matching points
        self.xyzm1 = self.mx1
        self.xyzm2 = self.mx2

        return self.pc_i
    
    
    
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

    @staticmethod
    def icp_residual_computer(p1, p2, n):
        icp_residual = np.sum(n*(p1 - p2), axis=1)
        return icp_residual
    







    
def icp_residual_for_pose(p1, p2, n, delta_T):
    p2_new = rigid_transform(p2, delta_T)
    residual = np.sum(n * (p2_new - p1), axis=1)
    return residual



def icp_error_func(p1,p2,n):
    def error_func(this, value, H):
        key = this.keys()[0]
        delta_pose = value.atPose3(key)
        delta_T = delta_pose.matrix()

        residual = icp_residual_for_pose(p1,p2,n,delta_T)

        if H is not None:
            H[0] = icp_error_jacobian(
                p2,
                n,
                delta_T
            )

        return residual
    return error_func
