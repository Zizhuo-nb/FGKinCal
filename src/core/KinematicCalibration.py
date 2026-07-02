# Imports
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
from colorama import init, Fore, Style
import shutil
import glob

# Import from src folder
from src.calibration.calibration import calibration
from src.calibration.kinematiccalibration import kinematiccalibration
from src.dataclasses.trajectory import Trajectory
from src.dataclasses.LMIdata import LaserdataLMI
from src.directgeoreferencing.directgeoreferencing import directgeoreferencing
from src.pointcloud.pointcloud import pointcloud
from src.config.sICPconfig import sICPconfig
from src.icp.SymPlane2PlaneICP import SymPlane2PlaneICP

# Import base functions
from src.base.base import RotmatX, RotmatY, RotmatZ, Rotmat2Euler, Euler2RotMat

class KinematicCalibration:

    # System calibration of the left and right scanner
    calL: calibration
    calR: calibration

    # Kinematic system calibration
    kcalL: kinematiccalibration
    kcalR: kinematiccalibration

    # Trajectory of the left and right scanner
    TL: Trajectory
    TR: Trajectory

    # LMI laser profiles of the left and right scanner
    lmidataL: LaserdataLMI
    lmidataR: LaserdataLMI

    # Config file of the sequential strip alignment
    config: sICPconfig

    xlist: list # List to store estimated parametes
    idxL: list
    idxR: list

    # Input & Output pathes
    parent_dir: str
    output_dir: str
    calibration_dir: str
    configfile: str

    def __init__(self, parent_dir, output_dir, calibration_dir, configfile):
        """
        """

        self.calL = calibration()
        self.calR = calibration()
        self.kcalL = kinematiccalibration()
        self.kcalR = kinematiccalibration()
        self.TL = Trajectory()
        self.TR = Trajectory()
        self.lmidataL = LaserdataLMI()
        self.lmidataR = LaserdataLMI()
        self.config = sICPconfig()
        self.xlist = []
        self.idxL = []
        self.idxR = []
        self.parent_dir = parent_dir
        self.output_dir = output_dir
        self.calibration_dir = calibration_dir
        self.configfile = configfile

    def print_info(self):
        """
        """

        print(( " ____________________________________________________________________________\n"
                "| \n"
               f"| {Style.BRIGHT}{Fore.MAGENTA}{'Kinematic ICP strip alignment'}{Style.RESET_ALL}\n"
               f"| ___________________________________________________________________________\n"
               f"|"  ))
            
        t = time.strftime("%H:%M:%S")
        print(( f"| {Style.BRIGHT}{Fore.GREEN}{t + ' Dataset info '}{Style.RESET_ALL}" ))
        print("| - path to data:   ", self.parent_dir)
        print("| - path to output: ", self.output_dir)
        print("| - path to calibration file: ", self.calibration_dir)
        print("| - path to config file: ", self.configfile)
        print("| ")

    def copy_data(self, plot_id, date):
        """Copy all .trj and .bin files from the pataset directory to local repo"""

        print(( " ____________________________________________________________________________\n"
                "| \n"
               f"| {Style.BRIGHT}{Fore.MAGENTA}{'Copy data from dataset directory'}{Style.RESET_ALL}\n"
               f"|"  ))

        # Clear input path from old files
        self.clear_input_path("input/")

        print("| copy files from dataset directory to local repo ...")

        # Create path to copy from
        full_dir = os.path.join(self.parent_dir, plot_id, date)

        # 1) Copy laserprofiles

        # Get filenames of the laser profiles files
        laser_dir = os.path.join(full_dir, "01_laserprofiles")
        lmi_l_str = [f for f in os.listdir(laser_dir) if f.endswith('l.bin')][0]
        lmi_r_str = [f for f in os.listdir(laser_dir) if f.endswith('r.bin')][0]

        shutil.copy2(os.path.join(laser_dir, lmi_l_str), "input/")
        shutil.copy2(os.path.join(laser_dir, lmi_r_str), "input/")

        # 2) Copy trajectories

        # Get filenames of the trajectories
        traj_dir = os.path.join(full_dir, "02_trajectory")
        trj_l_str = [f for f in os.listdir(traj_dir) if f.endswith('l.trj')][0]
        trj_r_str = [f for f in os.listdir(traj_dir) if f.endswith('r.trj')][0]

        shutil.copy2(os.path.join(traj_dir, trj_l_str), "input/")
        shutil.copy2(os.path.join(traj_dir, trj_r_str), "input/")
        

        print("| ")

    def clear_input_path(self, folder_path):
        """Delete all .trj and .bin files in the specified folder"""
        
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist")
            return
        
        # Find all .trj and .bin files
        trj_files = glob.glob(os.path.join(folder_path, "*.trj"))
        bin_files = glob.glob(os.path.join(folder_path, "*.bin"))
        
        all_files = trj_files + bin_files
        
        if not all_files:
            print("No .trj or .bin files found")
            return
        
        deleted_count = 0
        for file_path in all_files:
            try:
                os.remove(file_path)
                print(f"| Cleaned: {file_path}")
                deleted_count += 1
            except OSError as e:
                print(f"Error deleting {file_path}: {e}")
        
        print(f"| Total files deleted: {deleted_count}")


    def loaddata(self):
        """
        """   

        # Get filenames of the trajectories
        trj_l_str = [f for f in os.listdir("input/") if f.endswith('l.trj')][0]
        trj_r_str = [f for f in os.listdir("input/") if f.endswith('r.trj')][0]
        
        # Read trajectories
        self.TL.read_from_file( path_to_file = "input/" + trj_l_str, offset_xyz = self.config.txyz )
        self.TR.read_from_file( path_to_file = "input/" + trj_r_str, offset_xyz = self.config.txyz )
        
        # Get filenames of the laserprofiles
        lmi_l_str = [f for f in os.listdir("input/") if f.endswith('l.bin')][0]
        lmi_r_str = [f for f in os.listdir("input/") if f.endswith('r.bin')][0]

        # Read laser data
        self.lmidataL.readbin( "input/"+lmi_l_str )
        self.lmidataR.readbin( "input/"+lmi_r_str )
        
        # __________________________________________________________
        # Intersect and interpolate data
        #
        
        self.lmidataL.intersecting( self.TL.time )
        self.lmidataR.intersecting( self.TR.time )

        self.TL = self.TL.interpolate( self.lmidataL.timestamps, kind = "cubic")
        self.TR = self.TR.interpolate( self.lmidataR.timestamps, kind = "cubic")

    def loadconfig(self):
        """
        """

        self.config.readfromjson( self.configfile )

    def writeconfig(self):
        """
        """

        self.config.writeToJson( "config/sICPconfig.json" )

    def loadcalibration(self):
        """
        """

        self.calL.read_calibration_from_xml( self.calibration_dir + "system_config_lmi_l.xml" )
        self.calR.read_calibration_from_xml( self.calibration_dir + "system_config_lmi_r.xml" )

    def create_pointcloud(self, calibration = "static"):
        """
        """

        if calibration == "static": 
            call = self.calL
            calr = self.calR
        elif calibration == "kinematic":
            call = self.kcalL
            calr = self.kcalR 

        georefL = directgeoreferencing( trajectory = self.TL,
                                        laserlines = self.lmidataL,
                                        systemcalibration = call )
        
        georefR = directgeoreferencing( trajectory = self.TR,
                                        laserlines = self.lmidataR,
                                        systemcalibration = calr )    

        pcl = georefL.run( calibration=calibration )
        pcr = georefR.run( calibration=calibration )

        return pcl, pcr

    def run(self):
        """
        贰.1 
        作用就是对每一个局部窗口，先用静态外参生成左右局部点云，再转换到当前按窗口的平均body frame里面，然后左右点云icp，最后把icp得到的6Dof修正量和窗口中心时间保存到Px.txt里面
        """

        # Get intervals
        '''
        贰.2
        下面这个就是得到窗口索引，返回两个列表：
        左右scanner的每个窗口的起止index， 比如idxL[0] = [1000, 1900]

        '''
        idxL, idxR = self.get_alignment_intervals()

        # Initial guess parameter
        '''
        贰.3
        设置icp初始列表，也就是[0,0,0,0,0,0]
        后面如果icp成功就会替换这些0，用的是上一个窗口的值来替换作为初始值，反之就是继续保持0
        也就是说前一个窗口的icp结果其实是可以用于下一个窗口的icp的初始值的，可以更快收敛
        也就是和未来因子图的平滑因子是相关的，这里是上一个窗口是下一个窗口的初始值，而因子图可以变成用between factor作为约束使得相邻窗口不跳变

        '''
        x0 = np.zeros(6)

        # Loop over intervals and run ICP alignment
        '''
        贰.4
        遍历每一个窗口
        第 i 个窗口：
            取这一段左右数据
            生成局部点云
            做 ICP
            保存结果
        '''
        for i in range(len(idxL)):

            # ___________________________________________________________________________________
            # A) Create point clouds

            # Get trajectory and laserdata of the current interval
            '''
            得到窗口内的索引
            '''
            idxleft = np.arange(idxL[i][0],idxL[i][1])
            idxright = np.arange(idxR[i][0],idxR[i][1])

            TLi = self.TL.crop_by_index( idxleft ) #当前窗口的轨迹片段
            LMIl_i = self.lmidataL.crop_by_index( idxleft ) #当前窗口的laser profile片段

            TRi = self.TR.crop_by_index( idxright )
            LMIr_i = self.lmidataR.crop_by_index( idxright )

            # Run point cloud creation
            georefL = directgeoreferencing( TLi, LMIl_i, self.calL )
            pcl_i = georefL.run( calibration="static" ) #用静态外参生成点云，这里用的是static

            georefR = directgeoreferencing( TRi, LMIr_i, self.calR )
            pcr_i = georefR.run( calibration="static" )

            # ___________________________________________________________________________________
            # B) Transform point clouds into mean body frame of the current interval

            # Mean trajectory state of the interval
            '''
            找当前窗口中心的轨迹状态和时间
            '''
            idxmL = round((idxL[i][0] + idxL[i][1]) / 2) #中心index
            Tmil =  self.TL.statesall[idxmL, :] # 中心时间对应的轨迹pose
            timei = self.TL.time[idxmL] #对应的时间戳，后面保存到Px.txt的时间戳就是这个timei

            # Transform point clouds，这里就是前面georeferenced的点，取出来

            pc_l = pcl_i.xyz 
            pc_r = pcr_i.xyz

            # Translation，减去窗口中心位置，也就是把窗口中心移动到原点
            XG_l = (pc_l[:,0] - Tmil[1])
            YG_l = (pc_l[:,1] - Tmil[2]) 
            ZG_l = (pc_l[:,2] - Tmil[3]) 
                
            XG_r = (pc_r[:,0] - Tmil[1])
            YG_r = (pc_r[:,1] - Tmil[2]) 
            ZG_r = (pc_r[:,2] - Tmil[3])

            xyz_e_left = np.column_stack((XG_l, YG_l, ZG_l))
            xyz_e_right = np.column_stack((XG_r, YG_r, ZG_r))

            # Rotation，构造窗口中心的body的旋转矩阵
            R_B_NED_left = np.dot(np.dot(RotmatZ(Tmil[9]), RotmatY(Tmil[8])), RotmatX(Tmil[7]))
                
            # Transformation，进行旋转，icp不是在全局坐标做，而是在当前窗口的平均body frame下做
            '''
            为什么要在body frame下做？
            因为动态外参本来就是 scanner 相对于 body 的安装变化。
            如果直接在全局坐标系下 ICP，点云坐标会受到大地坐标、大尺度位置、轨迹方向的影响。
            变到当前窗口中心 body frame 后：

            '''
            pc_l = (R_B_NED_left.T @ xyz_e_left.T).T #这里是转回去
            pc_r = (R_B_NED_left.T @ xyz_e_right.T).T

            # ___________________________________________________________        
            # C) Run ICP on the point clouds

            print(( f"|______________________________________________________________________________________________________\n"
                    f"| {Style.BRIGHT}{Fore.MAGENTA}{'Running ICP alignment ('+str(i+1)+' /'+str(len(idxL))+')'}{Style.RESET_ALL} \n"
                    f"|  \n"
                    f"| - x0:  {Style.BRIGHT}{Fore.WHITE}{x0}{Style.RESET_ALL}\n"
                    f"| - number of point left:   {Style.BRIGHT}{Fore.WHITE}{len(pcl_i.xyz)}{Style.RESET_ALL}\n"
                    f"| - number of point right:  {Style.BRIGHT}{Fore.WHITE}{len(pcr_i.xyz)}{Style.RESET_ALL}"))
            
            # Set up ICP
            '''
            pc_l：当前窗口左 scanner 点云，已经在窗口中心 body frame 下
            pc_r：当前窗口右 scanner 点云，已经在窗口中心 body frame 下
            x0：ICP 初始值
            self.config：ICP 参数
            '''
            icp_instance = SymPlane2PlaneICP(pc_l, pc_r, x=x0)
            xi, suc = icp_instance.runICP( self.config )
            '''
            from factor_graph.core.icp_factor import icpFactor
            icp = icpFactor( pc_l, pc_r)
            
            pc_i = icp.matching(self.config)
            residual = icp.icpError()

            print("pc_l shape:", pc_l.shape)
            print("pc_r shape:", pc_r.shape)
            print("pc_i shape:", pc_i.shape)
            print("residual shape:", residual.shape)
            '''
            
            # Update initial guess for the next interval
            if suc == True:
                x0 = xi.copy()
            else:
                x0 = np.zeros(6)

            print(( f"| - Final transformation:  {Style.BRIGHT}{Fore.WHITE}{xi}{Style.RESET_ALL} "))
            print(( f"| - Number of point matches:  {Style.BRIGHT}{Fore.WHITE}{len(icp_instance.xyzm1)}{Style.RESET_ALL}"))
            
            # Store transformation parameters
            self.xlist.append({"left_indices": [idxL[i][0], idxL[i][1]],
                               "right_indices": [idxR[i][0], idxR[i][1]],
                               "transformation": xi.tolist(),
                               "timestamp": timei })
        
        # Write parameter correlation matrix to file
        with open(self.output_dir+"Px.txt", "w") as f:   
            for transformation in self.xlist:
                left_indices = ",".join(map(str, transformation['left_indices']))
                right_indices = ",".join(map(str, transformation['right_indices']))
                transformation_values = ",".join(map(str, transformation['transformation']))
                timestamp = transformation['timestamp']
                f.write(f"{left_indices},{right_indices},{transformation_values},{timestamp}\n")
        '''
        这段的本质：
        for 每一个局部窗口:
            1. 从左右 scanner 中取对应时间段的数据
            2. 用静态外参生成左右 georeferenced 点云
            3. 把两块点云变换到窗口中心 body frame
            4. 在 body frame 下做左右点云 ICP
            5. 得到该窗口的 6DoF 修正 xi
            6. 把 xi 和窗口中心时间保存下来
        '''

    def compute_kinematic_calibration_parameter(self):
        """
        壹.1：
        总的来说，icp得到的每个窗口的修正量Px.txt，转换成左右扫描仪各自随时间变化的动态外参，然后插值到完整轨迹的时间戳，并且写入文件。总的流程：
            1. 读取 ICP 结果 Px.txt
            2. 准备保存左右 scanner 动态外参的数组
            3. 读取左右 scanner 的静态外参，并转成齐次矩阵
            4. 遍历每个 ICP 窗口结果，把 ICP 修正分配给左右 scanner
            5. 对离散动态外参做插值，并写出文件

        """
        # Load icp parameter from file
        '''
        壹.2:
        这下面的两个函数
        读取的是前面run()生成的ICP结果文件PX.txt,每一行就是一个窗口的ICP对齐的结果。第二行代码是把含有NAN的行去掉。
        最后得到的icp_param就是一个矩阵，是N X 若干列，N是成功的ICP窗口的数量。
        '''
        icp_param = np.loadtxt( fname=self.output_dir+"Px.txt", delimiter="," )
        icp_param = icp_param[~np.isnan(icp_param).any(axis=1)]

        # Initialize kinematic calibration parameter
        '''
        壹.3：
        初始化左右动态外参的数组，分别是kcalL.x和kcalR.x，每个数组的大小是N X 7，每一行表示一个窗口中心时间上的外参[time, rx, ry, rz, tx, ty, tz]。
        就是准备了一个表格，用来储存 每个窗口中心时间上的左右扫描仪的外参
        '''
        self.kcalL.x = np.zeros((icp_param.shape[0], 7))
        self.kcalR.x = np.zeros((icp_param.shape[0], 7))

        # _____________________________________________________________________________
        # Get homogeneous transformation matrices for static calibration
        '''
        壹.4：
        把静态外参转换成齐次矩阵。
        里面的self.calL是左scanner的静态外参，里面包含：rx, ry, rz, tx, ty, tz。代码先用欧拉角生成旋转矩阵R_BS_L,然后再构造其次矩阵H_sbl。右scanner同理。
        最后得到的H_sbl和H_sbr就是左右scanner的静态外参的齐次矩阵表示。
        '''
        R_BS_L = RotmatZ( np.deg2rad(self.calL.rz) ) @ RotmatY( np.deg2rad(self.calL.ry) ) @ RotmatX( np.deg2rad(self.calL.rx) )
        H_sbl = self.create_homogeneous_matrix(R_BS_L.T, np.array((self.calL.tx, self.calL.ty, self.calL.tz))) 

        R_BS_R = RotmatZ( np.deg2rad(self.calR.rz) ) @ RotmatY( np.deg2rad(self.calR.ry) ) @ RotmatX( np.deg2rad(self.calR.rx) )
        H_sbr = self.create_homogeneous_matrix(R_BS_R.T, np.array((self.calR.tx, self.calR.ty, self.calR.tz)))

        # _____________________________________________________________________________
        # Get homogeneous transformation matrices for kinematic calibration
        '''
        壹.5：
        遍历每一个ICP窗口。每次循环处理一个ICP窗口，也就是：
            第i该窗口：
                有一个中心时间time_i
                有一个ICP修正xi_i
                要生成这个窗口上的左、右scanner的外参。
        '''

        for i in range(icp_param.shape[0]):

            # Get time stamp 读取窗口中心时间，icp_param每一行的最后一列是窗口中心时间，timei就是当前ICP对应的时间戳
            timei = icp_param[i,-1]

            # Translation and rotation estimated by ICP。 读取icp给出的旋转和平移，也就是在Px.txt里面，第 4,5,6 列：ICP 估计的旋转修正 rx, ry, rz，第 7,8,9 列：ICP 估计的平移修正 tx, ty, tz，最后一列：窗口中心时间。这里的R是把旋转角转换成旋转矩阵
            rx, ry, rz = icp_param[i, 4], icp_param[i, 5], icp_param[i, 6]
            R = Euler2RotMat( rx, ry, rz )
            t = np.array([icp_param[i,7], icp_param[i,8], icp_param[i,9]])

            # Transformation matrix最关键的地方。把ICP修正分成左右两半，两个参数关分别是表示两边的修正
            H_bbl = self.create_homogeneous_matrix(R.T, -t/2)
            H_bbr = self.create_homogeneous_matrix(R,    t/2)

            # __________________________________________________________
            # Kinematic calibration scanner left
            #

            # Update transformation。更新左scanner的外参。左scanner的新外参=左scanner ICP修正 X 左scanner静态外参
            H_sb_newl = H_bbl @ H_sbl

            # Compute euler angles，然后把旋转矩阵转回欧拉角
            rXrYrZ_l = Rotmat2Euler(H_sb_newl[:3, :3].T)

            # store kinematic calibration 最后存起来，也就是：
            # self.kcalL.x[i] = [
            #     当前窗口中心时间,
            #     左 scanner 新 rx,
            #     左 scanner 新 ry,
            #     左 scanner 新 rz,
            #     左 scanner 新 tx,
            #     左 scanner 新 ty,
            #     左 scanner 新 tz
            #     ]
            self.kcalL.x[i,0] = timei
            self.kcalL.x[i,1:4] = rXrYrZ_l
            self.kcalL.x[i,4:7] = H_sb_newl[:3, 3]

            # __________________________________________________________
            # Kinematic calibration scanner right
            #同理，最后得到的self.kcalL.x，self.kcalR.x里面保存的就是离散的外参序列

            # Update transformation
            H_sb_newr = H_bbr @ H_sbr

            # Compute euler angles
            rXrYrZ_r = Rotmat2Euler(H_sb_newr[:3, :3].T)

            # store kinematic calibration
            self.kcalR.x[i,0] = timei
            self.kcalR.x[i,1:4] = rXrYrZ_r
            self.kcalR.x[i,4:7] = H_sb_newr[:3, 3]

        # _____________________________________________________________________________
        # Interpolate kinematic calibration parameters
        #

        # Fill borders and interpolate using trajectory timestamps
        '''
        衔接上面，最后填充边界并且插值，下面四行代码，也就是从窗口中心时刻的离散外参，变成每个laser profile时间戳上的连续外参
        这里的fill_borders()解决的是，开头和结尾通常不会被ICP中心时间覆盖，这里把起点和终点也补上合理的外参值，然后interpolate_cubic_spline，插值给每一个时间戳
        '''
        self.kcalL.fill_borders( self.TL.time )
        self.kcalL.interpolate_cubic_spline( self.TL.time )

        self.kcalR.fill_borders( self.TR.time )
        self.kcalR.interpolate_cubic_spline( self.TR.time )

        # Visualize kinematic calibration parameter
        #self.kcalL.plot( self.calL )
        #self.kcalR.plot( self.calR )
        '''
        最后写出动态外参文件。
        '''
        self.kcalL.write_to_file(path_out=self.output_dir, fname="l")
        self.kcalR.write_to_file(path_out=self.output_dir, fname="r")

        '''
        目前函数的思想：
            每个窗口的 ICP 结果互相独立
            每个窗口直接生成一个外参修正
            窗口之间没有联合优化
            窗口之间的平滑只靠后面的 cubic spline 插值
        未来因子图：
            每个窗口中心时间建立一个外参节点 X_i
            ICP 不是直接生成最终外参，而是提供一个观测约束
            再加入静态外参 prior factor
            再加入相邻窗口 between/smoothness factor
            最后联合优化所有 X_i

            当前版本：
            Px.txt → 分配修正 → 插值 → 动态外参

            因子图版本：
            ICP residual/prior/smoothness → graph optimization → 外参节点序列 → 插值 → 动态外参
        '''
        

    def load_transformation_parameters(self, filename, left):

        df = pd.read_csv(filename, header=None)
        df.columns = ['left_start', 'left_end', 'right_start', 'right_end', 'param1', 'param2', 'param3', 'param4', 'param5', 'param6', 'timestamp']
        df = df[['left_start', 'left_end', 'right_start','right_end','param1', 'param2', 'param3', 'param4', 'param5', 'param6', 'timestamp']].to_numpy()

        """
        if left_right == "left":
            df_transformation = df_transformation[['left_start', 'left_end', 'param1', 'param2', 'param3', 'param4', 'param5', 'param6']]
            df_transformation = df_transformation.to_numpy()
        else:
            df_transformation = df_transformation[['right_start', 'right_end', 'param1', 'param2', 'param3', 'param4', 'param5', 'param6']].to_numpy()
        """

        # Initialize kinematic calibration
        self.kcalL.x = np.zeros((len(df), 7))
        self.kcalR.x = np.zeros((len(df), 7))

        print(df)

        return df
    
    def create_homogeneous_matrix(self, R, t):
        """
        Create a 4x4 homogeneous transformation matrix from a 3x3 rotation matrix and a 3x1 translation vector.
        
        Parameters:
        R: 3x3 rotation matrix
        t: 3x1 translation vector
        
        Returns:
        H: 4x4 homogeneous transformation matrix
        """
        # Check if R is a 3x3 matrix
        if R.shape != (3, 3):
            raise ValueError("Rotation matrix R must be 3x3.")
        
        # Check if t is a 3x1 vector
        if t.shape != (3,) and t.shape != (3, 1):
            raise ValueError("Translation vector t must be a 3x1 vector.")
        
        # Ensure t is a column vector
        t = t.reshape(3, 1)
        
        # Create the homogeneous transformation matrix
        H = np.eye(4)
        H[:3, :3] = R
        H[:3, 3] = t.flatten()
        
        return H

    def get_alignment_intervals(self):

        distance = 0
        index_end = None

        # Index offset where to start in the beginning
        idx0 = 0
        
        trajectory_length = self.TL.statesall.shape[0]
        all_indices = []
        index_start = 0

        while index_start < trajectory_length - 1:
            distance = 0
            for i in range(index_start, trajectory_length - 1):
                dist = math.sqrt((self.TL.statesall[i, 1] - self.TL.statesall[i+1, 1])**2 + (self.TL.statesall[i, 2] - self.TL.statesall[i+1, 2])**2)
                distance += dist

                if distance >= self.config.window_size:
                    index_end = i + 1
                    all_indices.append((index_start+idx0, index_end+idx0))
                    break

            # Find new index_start based on the step size
            distance = 0
            for j in range(index_start, trajectory_length - 1):
                dist = math.sqrt((self.TL.statesall[j, 1] - self.TL.statesall[j+1, 1])**2 + (self.TL.statesall[j, 2] - self.TL.statesall[j+1, 2])**2)
                distance += dist
                if distance >= self.config.step_size:
                    index_start = j + 1
                    break

            # If the step size loop completes without breaking, end the main loop
            else:
                break  

        # _____________________________________________________________
        # Find corresponding interval indices of the right trajectory
        #

        all_indices2 = []

        for i in range(len(all_indices)):
            idxA_t1, idxB_t1 = np.array(all_indices[i])

            timet1_A, timet1_B = self.TL.statesall[idxA_t1,0], self.TL.statesall[idxB_t1,0]

            idxA_t2 = np.argmin( np.abs(self.TR.statesall[:,0] - timet1_A ) )
            idxB_t2 = np.argmin( np.abs(self.TR.statesall[:,0] - timet1_B ) )

            all_indices2.append((idxA_t2, idxB_t2))

        return all_indices, all_indices2       