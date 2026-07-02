import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import time
import os
import pandas as pd
from colorama import init, Fore, Style


from src.base.base import find_duplicate

#import src.geodetictools.transformations as geobase

class Trajectory:
    def __init__(self, numberofstates: int = 0,
                       numberofbiasstates: int = 0,
                       time: int = [],
                       time_frame: str = [],
                       xyz_frame: str = [],
                       rpy_frame: str = [] ) -> None:
        
        # 1) General Trajectory Info
        self.numberofstates = numberofstates
        self.numberofbiasstates = numberofbiasstates
        self.time = time
        self.time_frame = time_frame
        self.xyz_frame = xyz_frame
        self.rpy_frame = rpy_frame

        # All states [time, xyz, velocity, rpy, bias acceleration, bias gyroscope] 16
        self.statesall = np.zeros( (numberofstates, 16) , dtype=float)
        self.bbox2D = np.array([0,0,0,0]) # Xmin, Xmax, Ymin, Ymax

    def read_from_file(self, path_to_file, dl=' ', offset_xyz = None ):
        
        t = time.strftime("%H:%M:%S")
        print(( f"| {Style.BRIGHT}{Fore.GREEN}{t + ' Read trajectory file from folder'}{Style.RESET_ALL}" ))
        print("| - path: ", path_to_file)

        self.statesall = np.loadtxt( path_to_file, delimiter=dl, comments="#")
        self.bbox2D = self.compute_bbox2D()
        self.time = self.statesall[:,0]

        # Apply offset
        if offset_xyz is not None:
            self.statesall[:,1] -= offset_xyz[0]
            self.statesall[:,2] -= offset_xyz[1]
            self.statesall[:,3] -= offset_xyz[2]

        print("| - first state: ", ' '.join(f"{num:.5f}" for num in self.statesall[0, :]))
        print("|  ")
        
    def interpolate_states(self):
        num_states = self.statesall.shape[0]
        
        # Interpolating xyz
        xyz = self.statesall[:, 1:4]
        for i in range(3):  # For x, y, z
            valid = ~np.isnan(xyz[:, i])
            interp = interp1d(np.where(valid)[0], xyz[valid, i], kind='linear', fill_value="extrapolate")
            xyz[:, i] = interp(np.arange(num_states))
        
        # Interpolating rpy using SLERP
        timestamps = self.statesall[:, 0]
        valid_rpy_mask = ~np.isnan(self.statesall[:, 7:10]).any(axis=1)
        valid_rpy_indices = np.where(valid_rpy_mask)[0]
        valid_rpy_timestamps = timestamps[valid_rpy_indices]
        
        # Euler Angles To Rotation structure        
        valid_r = R.from_euler('xyz', self.statesall[valid_rpy_indices, 7:10], degrees=False)
        
        # Slerp interpolation, as geodetic curve on unit sphere
        slerp = Slerp(valid_rpy_timestamps, valid_r)
        
        # Interpolated rotations
        interp_rots = slerp(timestamps)
        
        # Convert interpolated rotations back to Euler angles
        interpolated_rpy = interp_rots.as_euler('xyz', degrees=False)
        
        # Store the interpolated values back into the statesall array
        self.statesall[:, 1:4] = xyz
        self.statesall[:, 7:10] = interpolated_rpy

    def interpolate(self, timestamps, kind = "cubic"):

        # Find & delete duplicate values in time vector
        idx__ = find_duplicate( self.statesall[:,0] )
        self.statesall = np.delete( self.statesall, idx__, axis=0)
        self.numberofstates = len( self.statesall[:,0] )
        self.time = np.delete( self.time, idx__, axis=0)
        
        # Translation
        f_X = interp1d(self.statesall[:,0], self.statesall[:,1], kind=kind, fill_value="extrapolate") # X ECEF
        f_Y = interp1d(self.statesall[:,0], self.statesall[:,2], kind=kind, fill_value="extrapolate") # Y ECEF
        f_Z = interp1d(self.statesall[:,0], self.statesall[:,3], kind=kind, fill_value="extrapolate") # Z ECEF

        # find intersecting interval of data
        X_int = f_X( timestamps )
        Y_int = f_Y( timestamps )
        Z_int = f_Z( timestamps )

        # Velocity
        f_VX = interp1d(self.statesall[:,0], self.statesall[:,4], kind=kind, fill_value="extrapolate") # X velo
        f_VY = interp1d(self.statesall[:,0], self.statesall[:,5], kind=kind, fill_value="extrapolate") # Y velo
        f_VZ = interp1d(self.statesall[:,0], self.statesall[:,6], kind=kind, fill_value="extrapolate") # Z velo

        # find intersecting interval of data
        VX_int = f_VX( timestamps )
        VY_int = f_VY( timestamps )
        VZ_int = f_VZ( timestamps )

        # Euler Angles To Rotation structure 
        r = R.from_euler('xyz', np.c_[self.statesall[:,7], self.statesall[:,8], self.statesall[:,9]], degrees=False)
        
        # Slerp interpolation, as geodetic curve on unit sphere
        slerp = Slerp(self.statesall[:,0], r)

        interp_rots = slerp( timestamps )

        # Rotation to Euler Angles
        rpy_inter = interp_rots.as_euler('xyz', degrees=False)

        roll_int  = rpy_inter[:,0] # roll
        pitch_int = rpy_inter[:,1] # pitch
        yaw_int   = rpy_inter[:,2] # yaw

        # Fill new Trajectory with data
        Tr_interpolated = Trajectory()
        
        # All states [time, xyz, velocity, rpy, bias acceleration, bias gyroscope] 16
        Tr_interpolated.statesall = np.c_[timestamps, X_int, Y_int, Z_int, VX_int, VY_int, VZ_int, roll_int, pitch_int, yaw_int, np.zeros( (len(yaw_int),6)) ]
        
        Tr_interpolated.x=X_int
        Tr_interpolated.y=Y_int
        Tr_interpolated.z=Z_int
        Tr_interpolated.xyz_frame = self.xyz_frame
        Tr_interpolated.rpy_frame = self.rpy_frame

        # Trajectory Data
        Tr_interpolated.numberofstates = len(roll_int)
        Tr_interpolated.time = timestamps

        Tr_interpolated.bbox2D = np.array([np.min( Tr_interpolated.statesall[:,1] ),
                                           np.max( Tr_interpolated.statesall[:,1] ),
                                           np.min( Tr_interpolated.statesall[:,2] ),
                                           np.max( Tr_interpolated.statesall[:,2] )])

        return Tr_interpolated

    def crop_by_indices(self, start_idx, end_idx):

        # Create a new Trajectory3D object
        T_ = Trajectory()

        # Crop the trajectory data based on the provided start and end indices
        T_.statesall = self.statesall[start_idx:end_idx, :]  # Add 1 to include the end index
        
        # Extract time information from the cropped states
        T_.time = T_.statesall[:, 0]
        
        # Update the number of states in the cropped trajectory
        T_.numberofstates = end_idx - start_idx
        
        # Calculate the 2D bounding box of the cropped trajectory
        T_.bbox2D = np.array([np.min(T_.statesall[:, 1]),
                              np.max(T_.statesall[:, 1]),
                              np.min(T_.statesall[:, 2]),
                              np.max(T_.statesall[:, 2])])
        return T_
    
    def crop_by_index( self, idx ):

        T_ = Trajectory()

        T_.statesall = self.statesall[ idx,: ]
        T_.time = T_.statesall[:,0]
        T_.numberofstates = len( idx )

        T_.bbox2D = np.array([np.min( T_.statesall[:,1] ),
                              np.max( T_.statesall[:,1] ),
                              np.min( T_.statesall[:,2] ),
                              np.max( T_.statesall[:,2] )])
        return T_
    
    def compute_bbox2D(self):

        self.bbox2D = np.array([np.min( self.statesall[:,1] ),
                                np.max( self.statesall[:,1] ),
                                np.min( self.statesall[:,2] ),
                                np.max( self.statesall[:,2] )])
    
    def crop_by_index( self, idx ):

        T_ = Trajectory()

        T_.statesall = self.statesall[ idx,: ]
        T_.time = T_.statesall[:,0]
        T_.numberofstates = len( idx )

        T_.bbox2D = np.array([np.min( T_.statesall[:,1] ),
                              np.max( T_.statesall[:,1] ),
                              np.min( T_.statesall[:,2] ),
                              np.max( T_.statesall[:,2] )])

        return T_
    
    def write_to_file(self, path, filename, offsxyz = np.array([0,0,0])):
            
            # Create Folder
            if not os.path.exists( path ):
                os.makedirs( path )

            # add global offset
            self.statesall[:,1] += offsxyz[0]
            self.statesall[:,2] += offsxyz[1]
            self.statesall[:,3] += offsxyz[2]

            print( time.strftime("%H:%M:%S"), " write trajectory: #T = ", len(self.statesall[:,0]) )

            df = pd.DataFrame(data = self.statesall[0:-1,0:10])
            df.to_csv( path + filename + ".trj", sep=' ', header=False, float_format='%.8f', index=False)

            with open(path + filename + ".trj", 'r+') as file:
                content = file.read()
                file.seek(0, 0)  # Setze den Schreibkopf an den Anfang der Datei
                file.write("#name sbgekf\n")
                file.write("#epsg 25832\n")
                file.write("#fields t,px,py,pz,vx,vy,vz,ex,ey,ez\n")
                file.write(content)

            #name sbgekf
            #epsg 25832
            #fields t,px,py,pz,vx,vy,vz,ex,ey,ez

            # remove global offset
            self.statesall[:,1] -= offsxyz[0]
            self.statesall[:,2] -= offsxyz[1]
            self.statesall[:,3] -= offsxyz[2]
    
    def interpolate_cubic_spline(self, timestamps):

        retT = Trajectory()
        retT.statesall = np.zeros((timestamps.shape[0], 10))
        retT.statesall[:,0] = timestamps

        # extract ovseravtions for return
        mask = ~np.isnan(self.statesall[:,1])
        l = np.full((np.sum(mask), 7), np.nan)
        l[:,:4] = self.statesall[mask,:4]
        l[:,4:] = self.statesall[mask,7:10]     # [t,x,y,z,r,p,y]

        # fill borders -> better approximation an interpolation later
        xyz_median = np.median(l[:,1:4], axis=0)
        rpy_median = np.median(l[:,4:], axis=0)
        if np.isnan(self.statesall[0, 7:10]).any():
            self.statesall[0,1:4] = xyz_median
            self.statesall[0,7:10] = rpy_median
        if np.isnan(self.statesall[-1, 7:10]).any():
            self.statesall[-1,1:4] = xyz_median
            self.statesall[-1,7:10] = rpy_median

        # find filled items
        mask = ~np.isnan(self.statesall[:,1])
        l_trafo = np.full((np.sum(mask), 7), np.nan)
        l_trafo[:,:4] = self.statesall[mask,:4]
        l_trafo[:,4:] = self.statesall[mask,7:10]     # [t,x,y,z,r,p,y]

        # copute spline approximation (with Hermite Basis and Huber Estimator)
        border, param = self.approximate_Hermite_Huber(l_trafo)     # [x,x',y,y',z,z',r,r',p,p',y,y']

        # compute interpolated states
        xyz, rpy = self.evaluate_Hermite(border, param, timestamps)

        # fill self.statesall
        retT.statesall[:, 1:4] = xyz
        retT.statesall[:, 7:10] = rpy
        
        return retT

    def approximate_Hermite_Huber(self,l_trafo):

        # define borders
        num_states = l_trafo.shape[0]
        t_min = np.min(l_trafo[:,0])
        t_max = np.max(l_trafo[:,0])
        num_points = int(np.round(num_states / 6)) + 1
        border = np.linspace(t_min, t_max, num_points)
        
        # initilize param
        param = np.full((num_points, 12), np.nan)   # [x,x',y,y',z,z',r,r',p,p',y,y']
        # compute Design-Matrix
        A = self.getA(l_trafo[:,0],border)
        # compute params
        for i in range(6):      # i=0->x, i=1->y, ...
            l = l_trafo[:,i+1]
            # initilize result vector
            x = np.zeros(2*num_points)
            x[::2] = np.median(l)
            # initilize variables
            iter = 0
            dx = np.inf
            # compute parameters
            while np.max(np.abs(dx)) > 10e-4 and iter < 10:
                iter = iter+1
                # compute v
                v = l-A@x
                # copute weights
                sigma = 1.4826*np.median(np.abs(v-np.median(v)))
                v = v/sigma
                idx_0 = v==0
                v[idx_0] = 1
                w = self.Psi_Huber(v)/v
                # compute new x and dx
                x_old = x
                P = np.diag(w)
                x = np.linalg.inv(A.T@P@A)@A.T@P@l
                dx = x-x_old
            if (iter==10):
                print(f'Problem {i}')
            # fill param
            param[:,2*i] = x[::2]
            param[:,2*i+1] = x[1::2]    # [x,x',y,y',z,z',r,r',p,p',y,y']
        return border,param
    
    def getA(self,t,border):
        # initilize A
        A = np.zeros((len(t),len(border)*2))
        # iterate over all intervals
        for i in range(len(border)-1):
            idx_min = 2*i
            idx_max = 2*i+4
            t_min = border[i]
            t_max = border[i+1]
            dt = t_max-t_min
            # find timestemps
            if i == 0:
                idx_t = (t>=t_min) & (t<=t_max)
            else:
                idx_t = (t>t_min) & (t<=t_max)
            t_i = t[idx_t]-t_min
            # compute values of Hermite Basis
            A_i = np.zeros((np.sum(idx_t),4))
            A_i[:,0] = 1-3*(t_i/dt)**2+2*(t_i/dt)**3
            A_i[:,1] = t_i*(1-2*(t_i/dt)+(t_i/dt)**2)
            A_i[:,2] = 3*(t_i/dt)**2-2*(t_i/dt)**3
            A_i[:,3] = t_i*(-(t_i/dt)+(t_i/dt)**2)
            # fill in A
            A[idx_t,idx_min:idx_max] = A_i
        return A

    def Psi_Huber(self,v, k=2):
        idx = np.abs(v)>k
        v[idx] = np.sign(v[idx])*k
        return v

    def evaluate_Hermite(self,border,param,t):
        # compute A
        A = self.getA(t,border)
        # compute values
        xyz = np.zeros((len(t),3))
        rpy = np.zeros((len(t),3))
        for i in range(3):
            x_xyz = np.zeros(2*len(border))
            x_xyz[::2] = param[:,2*i]
            x_xyz[1::2] = param[:,2*i+1]
            x_rpy = np.zeros(2*len(border))
            x_rpy[::2] = param[:,2*i+6]
            x_rpy[1::2] = param[:,2*i+7]
            xyz[:,i] = A@x_xyz
            rpy[:,i] = A@x_rpy
        return xyz, rpy
    


