import numpy as np
from sklearn.neighbors import NearestNeighbors
#from ..geodetictools.transformations import Euler2RotMat, Rotmat2Euler
from src.config.sICPconfig import sICPconfig
import random

from colorama import init, Fore, Style

class SymPlane2PlaneICP:
    def __init__(self, pc1, pc2, x=np.zeros(6), max_iter=50, dmax=0.06):
        self.pc1 = pc1              # point cloud 1
        self.pc2 = pc2              # point cloud 2
        self.pc_i = None            # point matches 1 & 2 + averaged normal vector
        self.xyzm1 = None           # point matches 1
        self.xyzm2 = None           # point matches 2
        self.x = x                  # parameter vector
        self.dx = None              # update on the parameter vector
        self.max_iter = max_iter    # maximum iterations of the ICP
        self.dmax = dmax            # threshold ICP convergence
        self.iterations = 0         # iteration counter

    def runICP(self, config: sICPconfig ):
        """Run the ICP algorithm

        Args:
        - config: Configfile for the ICP algorithm

        Return:
        - x: parameter vector of the transformation between the point clouds
        - suc: boolean variable to return the success of the optimization
        """

        '''
        叁.1
        函数的总体作用：
            给定两个窗口点云 pc1 和 pc2，从初始值 self.x 出发，反复做“匹配点 → 估计小修正 → 更新点云”，直到收敛，最后输出一个 6DoF 变换 x。
        函数头：
            self.pc1      左点云，已经在窗口中心 body frame 下
            self.pc2      右点云，已经在窗口中心 body frame 下
            self.x        ICP 初始值，也就是 x0
            config        ICP 配置参数

        输出：
            self.x = 最终 ICP 估计出的 6DoF 修正或者失败时返回nan
        '''

        # Determine homogenous transformation from initial guess parameter

        '''
        叁.2
        根据初始值构造左右修正矩阵，和前面compute_kinematic_calibration_parameter() 的思想一样，左右各修正一半

        '''
        r, p, y, t1, t2, t3 = self.x
        R = self.Euler2RotMat(r, p, y)
        t = np.array([t1 / 2, t2 / 2, t3 / 2])
        Hl = self.create_homogeneous_matrix(R.T, -t)
        Hr = self.create_homogeneous_matrix(R, t)

        # Apply initial transformation to point clouds 
        '''
        叁.3
        然后把修正量各自给左右点云进行初始修正，然后开始迭代
        '''
        self.pc1 = self.transform_points(self.x, self.pc1, "left")
        self.pc2 = self.transform_points(self.x, self.pc2, "right")

        for iter in range( config.max_iterations ):

            # Compute point-to-point matches based on distance
            '''
            叁.4
            寻找匹配点，在已经变换了的pc1与pc2之间寻找匹配点
            '''
            self.matching( config )

            # Estimate transformation with matched points   
            '''
            叁.5 根据匹配点估计小修正Δx
            ''' 
            self.estimateTrafo()
            
            # Apply transformation to point clouds
            '''
            叁.5
            把小修正作用到点云，也就是对称更新，左右各一半，也叫symmetric icp
            '''
            self.pc1 = self.transform_points(self.dx, self.pc1, "left")
            self.pc2 = self.transform_points(self.dx, self.pc2, "right")

            # Apply transformation to point matches
            ''''
            叁.6
            完整点云更新了，匹配的子点也更新一下
            '''
            self.xyzm1 = self.transform_points(self.dx, self.xyzm1, "left")
            self.xyzm2 = self.transform_points(self.dx, self.xyzm2, "right")

            # Update transformation matrices
            R = self.Euler2RotMat(self.dx[0], self.dx[1], self.dx[2])
            t = self.dx[3:] / 2
            dHl = self.create_homogeneous_matrix(R.T, -t)
            dHr = self.create_homogeneous_matrix(R, t)
            Hl = dHl @ Hl
            Hr = dHr @ Hr

            # Update iteration counter
            self.iterations += 1

            if self.dx is not None: 
                print(f"| - Iteration: {self.iterations}, dx = {[f'{x:.5f}' for x in self.dx]}")

            # Check threshold
            if max(abs(self.dx)) <= config.convergence_threshold:
                print(( f"| - {Style.BRIGHT}{Fore.WHITE}{'Number of iterations until convergence: '+str(iter + 1)}{Style.RESET_ALL}"))
                self.x = self.extract_parameters(Hr)
                return self.x, True

        # If not converged
        print(( f"| - {Style.BRIGHT}{Fore.RED}{'ICP did not converge after iteration: ' + str(self.max_iter)}{Style.RESET_ALL}"))
        self.x[:] = np.nan * np.ones(6)
        return self.x, False

    def matching(self, config: sICPconfig ):
        """Finds point matches between two point clouds
        
        Args:
        - config: Configfile for the ICP algorithm
        
        Return:
        - 
        """

        # Voxeldownsampling of the first point cloud
        pc1_downsampled, _ = self.voxel_downsampling(self.pc1, config.voxel_size)

        # Compute nearest neighbor distances to voxelized point cloud
        '''
        dNN    每个左点到最近右点的距离
        idxNN  每个最近右点在 self.pc2 里面的 index
        '''
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(self.pc2)
        dNN, idxNN = nbrs.kneighbors(pc1_downsampled)

        # Filter matches by distances threshold 如果最近邻距离太大，就认为这个匹配不可靠，丢掉
        valid_mask = dNN[:, 0] < config.max_dist

        # Store point matches idx 
        self.mx1 = pc1_downsampled[valid_mask]
        self.mx2 = self.pc2[idxNN[valid_mask, 0]]
        self.pc_i = np.hstack((self.mx1, self.mx2))

        # Compute normals and roughness for pc1, return: [nx, ny, nz, roughness, idx_valid] 
        '''
        对于每个左匹配点 self.mx1，
        在左完整点云 self.pc1 周围找邻域点，
        用邻域点拟合局部平面，
        得到法向量 n1 和 roughness/std1

        n1    左匹配点的局部法向量
        std1  局部平面的粗糙度/平面拟合残差
        idx   哪些点成功计算了法向量
        '''
        n1, std1, idx = SymPlane2PlaneICP.normals(self.mx1 , self.pc1, config.normals_radius, config.normals_minpoints, config.normals_maxpoints) 
        self.mx1 , self.mx2 = self.mx1 [idx], self.mx2[idx] 

        # Filter by roughness
        if config.roughness_filter_use:
            idx = std1 <= config.max_roughness
            self.mx1, self.mx2, n1 = self.mx1[idx], self.mx2[idx], n1[idx]

        # Compute normals and roughness for pc2
        
        n2, std2, idx = SymPlane2PlaneICP.normals(self.mx2, self.pc2, config.normals_radius, config.normals_minpoints, config.normals_maxpoints) #n: [nx, ny, nz, roughness]
        self.mx1, self.mx2, n1 = self.mx1 [idx], self.mx2[idx], n1[idx]

        # Filter by roughness value  如果启用了 roughness filter，那么会丢掉局部表面太粗糙的点。
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

        '''
        1. 左点云体素降采样
        2. 对每个左降采样点，在右点云中找最近邻
        3. 用最近邻距离过滤
        4. 对左匹配点估计法向量和 roughness
        5. 用 roughness 过滤左侧不稳定点
        6. 对右匹配点估计法向量和 roughness
        7. 用 roughness 过滤右侧不稳定点
        8. 检查左右法向量夹角，过滤不平行的点对
        9. 统一法向量方向
        10. 平均左右法向量
        11. 用 point-to-plane MAD 过滤离群匹配
        12. 生成 self.pc_i = [mx1, mx2, mean_normal]
        '''

    def estimateTrafo(self):
        """Estimate the rigid body transformation from matched points."""

        ''''
        肆.1
        输入是下面三行，分别是：
            x1       匹配点对中的左点
            x2       匹配点对中的右点
            normals  对应的法向量

        这里要解的是dx这个修正量
        '''

        x1 = self.pc_i[:,:3]
        x2 = self.pc_i[:,3:6]
        normals = self.pc_i[:,6:]

        # Estimate rigid body transformaton
        '''
        这里的残差l，解的是左右点的差在法向上的距离
        A矩阵就是雅可比，每个匹配点贡献一行：
            A_i = [关于 rx 的系数, 关于 ry 的系数, 关于 rz 的系数, 关于 tx 的系数, 关于 ty 的系数, 关于 tz 的系数]
        '''
        A = np.zeros([x1.shape[0],6])
        A [:,0] = -normals[:,1]*x2[:,2]+normals[:,2]*x2[:,1]-normals[:,1]*x1[:,2]+normals[:,2]*x1[:,1]
        A [:,1] =  normals[:,0]*x2[:,2]-normals[:,2]*x2[:,0]+normals[:,0]*x1[:,2]-normals[:,2]*x1[:,0]
        A [:,2] = -normals[:,0]*x2[:,1]+normals[:,1]*x2[:,0]-normals[:,0]*x1[:,1]+normals[:,1]*x1[:,0]
        A [:,3:] = normals
        l = normals[:,0]*(x1[:,0]-x2[:,0])+normals[:,1]*(x1[:,1]-x2[:,1])+normals[:,2]*(x1[:,2]-x2[:,2])

        # Huber weighting with 
        sigma = 1.4826*np.median(np.abs(np.abs(l)-np.median(np.abs(l))))

        v = l/sigma

        # Recude by median
        v = np.abs(v) - np.median(np.abs(v))

        idx_0 = v==0
        v[idx_0] = 1
        w = self.Psi_Huber(v)/v
        A = A*w[:, np.newaxis]
        l = l*w

        # Compute parameter updates
        dx = np.linalg.lstsq(A, l, rcond=None)
        self.dx = dx[0]

        # Compute parameter cofactor and correlation matrix 
        Qdxdx = np.linalg.inv(A.T @ A)
        Cdxdx = self.compute_correlation_matrix(Qdxdx)

        # Print correlation matrix to terminal
        #np.set_printoptions(precision=4, suppress=True)
        #print(Cdxdx)
        #np.set_printoptions()

    """ Basic transformations """
    @staticmethod
    def Euler2RotMat(roll, pitch, yaw):
        R = SymPlane2PlaneICP.RotmatZ( yaw ) @ SymPlane2PlaneICP.RotmatY( pitch ) @ SymPlane2PlaneICP.RotmatX( roll )
        return R
    
    @staticmethod
    def Rotmat2Euler( rotmat ):
        rX  = np.arctan2( rotmat[2,1], rotmat[2,2] )
        rY = np.arctan2( -rotmat[2,0], np.sqrt(rotmat[2,1]**2 + rotmat[2,2]**2) )
        rZ   = np.arctan2( rotmat[1,0], rotmat[0,0] )
        return np.array( [rX, rY, rZ] )
    
    @staticmethod
    def RotmatX(alpha):
        return np.array([ [1,0,0] , [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)] ])
    
    @staticmethod
    def RotmatY(beta):
        return np.array([ [np.cos(beta), 0, np.sin(beta) ], [0,1,0], [-np.sin(beta), 0, np.cos(beta)] ])
    
    @staticmethod
    def RotmatZ(gamma):
        return np.array([ [np.cos(gamma),-np.sin(gamma), 0] , [np.sin(gamma), np.cos(gamma), 0] , [0,0,1] ])
    
    @staticmethod
    def compute_correlation_matrix( Q ):
        """ Computes the parameter correlation matrix from a input cofactor matrix

        Args:
            Q: cofactor matrix [6x6]

        Returns:
            coor_matrix: correlation matrix [6x6]
        """
        std_devs = np.sqrt(np.diag(Q))
        corr_matrix = Q / np.outer(std_devs, std_devs)
        return corr_matrix

    @staticmethod
    def filter_invalid_points(pc):       
        """ filters invalid points of a point cloud

        Args:
            pc: input point cloud as numpy array [Nx3]

        Returns:
            pc: point cloud as numpy array [Mx3]
        """
        valid_mask = np.all(np.isfinite(pc), axis=1)
        return pc[valid_mask]

    @staticmethod
    def extract_parameters(Hr):
        """ Sets up the parameter vector from an input transformation matrix

        Args:
            Hr: transformation matrix 4x4

        Returns:
            x: parametervector [rx,ry,rz,tx,ty,tz]
        """
        x = np.zeros(6)
        x[:3] = SymPlane2PlaneICP.Rotmat2Euler(Hr[:3, :3])
        x[3:] = 2 * Hr[:3, 3] #这里最终输出的是完整相对变换
        return x
    
    @staticmethod
    def create_homogeneous_matrix(R, t):
        """Sets up a 4x4 transformation matrix for the input rotation matrix and translation vector.

        Args:
            R: Rotation matrix 3x3 
            t: translation vector 3x1

        Returns:
            T: Transformation matrix 4x4
        """
         
        if R.shape != (3, 3):
            raise ValueError("Rotation matrix R must be 3x3.")
        
        if t.shape != (3,) and t.shape != (3, 1):
            raise ValueError("Translation vector t must be a 3x1 vector.")
        
        t = t.reshape(3, 1)
        
        H = np.eye(4)
        H[:3, :3] = R
        H[:3, 3] = t.flatten()
        
        return H
    
    @staticmethod
    def transform_points(x_trafo, X, left_right):
        """Splits and applies the ICP transformation parameters to two point clouds.

        Args:
            x_trafo : r, p, y, t1,, t2, t3 from ICP 
            X : Old Point from point cloud
            left_right : either left or right scanner

        Returns:
            x : newly transformed point
        """
        r, p, y, t1, t2, t3 =  x_trafo

        R = SymPlane2PlaneICP.Euler2RotMat(r, p, y)
        translation = np.array([t1 / 2, t2 / 2, t3 / 2])
        
        if left_right == "right":
            x = R@X.T + translation[:, np.newaxis]
        else:
            x = R.T@X.T - translation[:, np.newaxis]
        
        return x.T
    
    @staticmethod
    def Psi_Huber(v, k=2):
        """ Computes the Huber weights for an input residual vector.

        Args:
            v: residuals
            k:

        Returns:
            v : weighted residuals
        """
        idx = np.abs(v)>k
        v[idx] = np.sign(v[idx])*k
        return v
        
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

                plane_normal, std_dev = SymPlane2PlaneICP.plane_fitting(neighbors)
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
    def voxel_downsampling(points, voxel_size):
        """ Computes the plane parameters for input points

        Args:
            points: 3D points as numps array

        Returns:
            points: downsampled points
            unique_indices: indices of the points in the input point cloud
        """
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)
        _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
        return points[unique_indices], unique_indices