from factor_graph.core.icp_factor import icpFactor
from src.core.KinematicCalibration import KinematicCalibration
from src.config.sICPconfig import sICPconfig
from factor_graph.core.prior_factor import prior_error
# from factor_graph.core.smooth_factor import smooth_factor
from src.directgeoreferencing.directgeoreferencing import directgeoreferencing
from src.base.base import RotmatX, RotmatY, RotmatZ, Rotmat2Euler, Euler2RotMat
from factor_graph.core.icp_factor import icpFactor
from factor_graph.tool.tool import rigid_transform
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from factor_graph.core.gtsam_optimizer import gtsam_optimize_sliding_icp
import numpy as np
import os
import glob
import shutil




class FactorGraph:
    config: sICPconfig
    def __init__(self,
                 parent_dir,
                 output_dir,
                 calibration_dir,
                 configfile,
                 plot_id,
                 date):
        self.parent_dir = parent_dir
        self.output_dir = output_dir
        self.calibration_dir = calibration_dir
        self.configfile = configfile
        self.plot_id = plot_id
        self.date = date
        self.icp_residual = None
        self.Delta_T_prev = np.eye(4)
        self.result = []

        self.window_buffer = []
        self.max_window_size = 5

        self.noise_model = {
            "icp_sigma": 1,
            "prior_sigmas": np.array([1, 1, 1, 1, 1, ]),
            "smooth_sigmas": np.array([1, 1, 1, 1, 1, 1])
        }


        self.config = sICPconfig()


    def run(self):
        kin_cal = KinematicCalibration(self.parent_dir,
                                       self.output_dir,
                                       self.calibration_dir,
                                       self.configfile)
        
        kin_cal.copy_data(self.plot_id , self.date)
        kin_cal.print_info()
        kin_cal.loadconfig()
        kin_cal.loadcalibration()
        kin_cal.loaddata()

        idxL, idxR = kin_cal.get_alignment_intervals()#same as before, generating windows for "ICP"
        for i in range(len(idxL)):
            pc_l, pc_r, timei = self.window_data(
                i = i,
                kin_cal = kin_cal,
                idxL= idxL,
                idxR= idxR
            )


            if i == 0:
                pc_r_for_icp = pc_r
            else:
                pc_r_for_icp = rigid_transform(pc_r, self.Delta_T_prev)

            p1, p2, n, icp_residual = self.factor_generator(pc_l, pc_r_for_icp) #using result from last step to initilize the right point cloud position
            
           
            item = {
                "window": i,
                "time": timei,
                "p1": p1,
                "p2": p2,
                "n": n,
                "icp_residual": icp_residual,
                "Delta_T_base": self.Delta_T_prev.copy(),
                "delta_T_init": np.eye(4),
                "delta_T_opt": None,
                "Delta_T": None,
            }
            self.window_buffer.append(item)
            if len(self.window_buffer) > 5:
                self.window_buffer.pop(0)


            self.window_buffer, result = gtsam_optimize_sliding_icp(self.window_buffer, self.noise_model)
            current_item = self.window_buffer[-1]

            Delta_T_i = current_item["Delta_T"]
            delta_T_opt = current_item["delta_T_opt"]

            self.Delta_T_prev = Delta_T_i


            self.result.append({
                "window": current_item["window"] + 1,
                "time": current_item["time"],
                "Delta_T": Delta_T_i,
                "num_matches": current_item["p1"].shape[0],
                "icp_rmse_before": np.sqrt(np.mean(current_item["icp_residual"] ** 2))
            })
           
            #===========
            p2_after = rigid_transform(p2, delta_T_opt)
            icp_residual_after = np.sum(n * (p1 - p2_after), axis=1)

            icp_rmse_before = np.sqrt(np.mean(icp_residual ** 2))
            icp_rmse_after = np.sqrt(np.mean(icp_residual_after ** 2))
            #===========
            print("window:", i + 1)
            print("time:", timei)
            print("matches:", p1.shape[0])
            print("icp rmse before:", icp_rmse_before)
            print("icp rmse after :", icp_rmse_after)
            print("Delta_T_i:")
            print(Delta_T_i)
            print("==========================")


        self.export_factor_graph_calibration_to_kincal(kin_cal)


        pcl, pcr = kin_cal.create_pointcloud(calibration="kinematic")

        # pc = pcl.concatenate(pcr) #if u want full point cloud

        pcl.write_to_file(
            path=kin_cal.output_dir,
            filename="pcl_factor_icpOnly_calibration",
            offset=kin_cal.config.txyz
        )

        pcr.write_to_file(
            path=kin_cal.output_dir,
            filename="pcr_factor_icpOnly_calibration",
            offset=kin_cal.config.txyz
        )
            
         
         
            

        # ===============================================        

    def factor_generator(self, pc_L, pc_R):
        
        icp_fac = icpFactor(pc_L,pc_R)
        pc_matching = icp_fac.matching(self.config) 
        
        if pc_matching is None or pc_matching.shape[0] == 0:
            raise ValueError("====icp residual is None====")
        
        p1 = pc_matching[:, 0:3]
        p2 = pc_matching[:, 3:6]
        n = pc_matching[:,6:9]
        icp_residual = icp_fac.icp_residual_computer(p1,p2,n)

        self.icp_residual = icp_residual
        print("number of matched points:", p1.shape ,"and", p2.shape)

        return p1, p2, n, icp_residual



    def window_data(self,i,kin_cal,idxL,idxR):
    #==================testing======================
        idxleft= np.arange(idxL[i][0],idxL[i][1])
        idxright= np.arange(idxR[i][0], idxR[i][1])

        TLi= kin_cal.TL.crop_by_index(idxleft) 
        LMIl_i = kin_cal.lmidataL.crop_by_index(idxleft)
        TRi= kin_cal.TR.crop_by_index(idxright)
        LMIr_i = kin_cal.lmidataR.crop_by_index(idxright)

        # Run point cloud creation
        georefL = directgeoreferencing( TLi, LMIl_i, kin_cal.calL )
        pcl_i = georefL.run( calibration="static" ) 

        georefR = directgeoreferencing( TRi, LMIr_i, kin_cal.calR )
        pcr_i = georefR.run( calibration="static" )


        # Mean trajectory state of the interval
   
        idxmL = round((idxL[i][0] + idxL[i][1]) / 2) 
        Tmil =  kin_cal.TL.statesall[idxmL, :] 
        timei = kin_cal.TL.time[idxmL] 

     

        pc_l = pcl_i.xyz 
        pc_r = pcr_i.xyz

        XG_l = (pc_l[:,0] - Tmil[1]) 
        YG_l = (pc_l[:,1] - Tmil[2]) 
        ZG_l = (pc_l[:,2] - Tmil[3]) 
            
        XG_r = (pc_r[:,0] - Tmil[1])
        YG_r = (pc_r[:,1] - Tmil[2]) 
        ZG_r = (pc_r[:,2] - Tmil[3])

        xyz_e_left = np.column_stack((XG_l, YG_l, ZG_l))
        xyz_e_right = np.column_stack((XG_r, YG_r, ZG_r))

        
        R_B_NED_left = np.dot(np.dot(RotmatZ(Tmil[9]), RotmatY(Tmil[8])), RotmatX(Tmil[7]))
            
        # Transformation
       
        pc_l = (R_B_NED_left.T @ xyz_e_left.T).T 
        pc_r = (R_B_NED_left.T @ xyz_e_right.T).T

        return pc_l, pc_r, timei
    

    def export_factor_graph_calibration_to_kincal(self, kin_cal):
   

        n = len(self.result)

        kin_cal.kcalL.x = np.zeros((n, 7))
        kin_cal.kcalR.x = np.zeros((n, 7))

        # 1. static left calibration -> homogeneous matrix
        R_BS_L = (
            RotmatZ(np.deg2rad(kin_cal.calL.rz))
            @ RotmatY(np.deg2rad(kin_cal.calL.ry))
            @ RotmatX(np.deg2rad(kin_cal.calL.rx))
        )

        H_sbl = kin_cal.create_homogeneous_matrix(
            R_BS_L.T,
            np.array([kin_cal.calL.tx, kin_cal.calL.ty, kin_cal.calL.tz])
        )

        # 2. static right calibration -> homogeneous matrix
        R_BS_R = (
            RotmatZ(np.deg2rad(kin_cal.calR.rz))
            @ RotmatY(np.deg2rad(kin_cal.calR.ry))
            @ RotmatX(np.deg2rad(kin_cal.calR.rx))
        )

        H_sbr = kin_cal.create_homogeneous_matrix(
            R_BS_R.T,
            np.array([kin_cal.calR.tx, kin_cal.calR.ty, kin_cal.calR.tz])
        )

       
        for i, item in enumerate(self.result):
            timei = item["time"]
            Delta_T_i = item["Delta_T"]

            H_sb_newl = H_sbl

    
            H_sb_newr = Delta_T_i @ H_sbr

            # left
            kin_cal.kcalL.x[i, 0] = timei
            kin_cal.kcalL.x[i, 1:4] = Rotmat2Euler(H_sb_newl[:3, :3].T)
            kin_cal.kcalL.x[i, 4:7] = H_sb_newl[:3, 3]

            # right
            kin_cal.kcalR.x[i, 0] = timei
            kin_cal.kcalR.x[i, 1:4] = Rotmat2Euler(H_sb_newr[:3, :3].T)
            kin_cal.kcalR.x[i, 4:7] = H_sb_newr[:3, 3]

     
        kin_cal.kcalL.xint = np.zeros((len(kin_cal.TL.time), 7))
        kin_cal.kcalL.xint[:, 0] = kin_cal.TL.time

        kin_cal.kcalL.xint[:, 1:4] = Rotmat2Euler(H_sbl[:3, :3].T)
        kin_cal.kcalL.xint[:, 4:7] = H_sbl[:3, 3]

        kin_cal.kcalR.fill_borders(kin_cal.TR.time)
        kin_cal.kcalR.interpolate_cubic_spline(kin_cal.TR.time)

 
        kin_cal.kcalL.write_to_file(path_out=kin_cal.output_dir, fname="l")
        kin_cal.kcalR.write_to_file(path_out=kin_cal.output_dir, fname="r")

   