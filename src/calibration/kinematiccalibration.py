import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class kinematiccalibration:

    x: np.ndarray
    xint: np.ndarray

    def __init__(self, x = np.ndarray, xint = np.ndarray):
        
        # kinematic calibration parameter 
        self.x = x # [t, rx, ry, rz, x, y, z]
        self.xint = xint # [t, rx, ry, rz, x, y, z]

    def fill_borders(self, timestamps):
        '''
        用中位数来补
        '''
        # Rotation
        rx_m, ry_m, rz_m = np.median(self.x[:,1]), np.median(self.x[:,2]), np.median(self.x[:,3])

        # Translation
        tx_m, ty_m, tz_m = np.median(self.x[:,4]), np.median(self.x[:,5]), np.median(self.x[:,6])
        
        self.x = np.insert(self.x, 0, np.c_[timestamps[0], rx_m, ry_m, rz_m, tx_m, ty_m, tz_m], axis=0)
        self.x = np.insert(self.x, self.x.shape[0], np.c_[timestamps[-1], rx_m, ry_m, rz_m, tx_m, ty_m, tz_m], axis=0)

    def write_to_file(self, path_out, fname):

        np.savetxt( fname=path_out+"x_"    +fname+".txt", X=self.x )
        np.savetxt( fname=path_out+"xint_" +fname+".txt", X=self.xint )

    def interpolate(self, timestamps):

        # Rotation
        f_rx = interp1d(self.x[:,0], self.x[:,1], kind="cubic", fill_value="extrapolate") 
        f_ry = interp1d(self.x[:,0], self.x[:,2], kind="cubic", fill_value="extrapolate") 
        f_rz = interp1d(self.x[:,0], self.x[:,3], kind="cubic", fill_value="extrapolate") 
        
        rx_int = f_rx( timestamps )
        ry_int = f_ry( timestamps )
        rz_int = f_rz( timestamps )

        # Translation
        f_tx = interp1d(self.x[:,0], self.x[:,4], kind="cubic", fill_value="extrapolate") 
        f_ty = interp1d(self.x[:,0], self.x[:,5], kind="cubic", fill_value="extrapolate") 
        f_tz = interp1d(self.x[:,0], self.x[:,6], kind="cubic", fill_value="extrapolate") 

        tx_int = f_tx( timestamps )
        ty_int = f_ty( timestamps )
        tz_int = f_tz( timestamps )

        self.xint = np.c_[timestamps, rx_int, ry_int, rz_int, tx_int, ty_int, tz_int ]

    def interpolate_cubic_spline(self, timestamps):

        # Observations
        l = np.c_[self.x[:,0], self.x[:,4], self.x[:,5], self.x[:,6], self.x[:,1], self.x[:,2], self.x[:,3]]

        # Compute spline interpolation (with Hermite Basis and Huber Estimator)
        border, param = self.approximate_Hermite_Huber(l) # [x,x',y,y',z,z',r,r',p,p',y,y']

        # compute interpolated states
        xyz, rpy, i0xyz, i0rpy = self.evaluate_Hermite(border, param, timestamps)

        # Write interpolated calibration parameter to file
        self.xint = np.c_[timestamps[i0xyz], rpy[i0rpy], xyz[i0xyz]]

    def approximate_Hermite_Huber(self,l_trafo):
        
        # define borders
        num_states = l_trafo.shape[0]
        t_min = np.min(l_trafo[:,0])
        t_max = np.max(l_trafo[:,0])
        num_points = int(np.round(num_states / 6)) + 1
        border = np.linspace(t_min, t_max, num_points)
        
        # initilize param
        param = np.full((num_points, 12), np.nan) # [x,x',y,y',z,z',r,r',p,p',y,y']
        # compute Design-Matrix
        A = self.getA(l_trafo[:,0],border)
        # compute params
        for i in range(6): # i=0->x, i=1->y, ...
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
            param[:,2*i+1] = x[1::2] # [x,x',y,y',z,z',r,r',p,p',y,y']

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

        # Compute A
        A = self.getA(t,border)

        # Compute values
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

        # Fill borders with median
        xyz_med = np.median(xyz, axis=0)
        rpy_med = np.median(rpy, axis=0)

        xyz[0,0] = xyz_med[0]
        xyz[0,1] = xyz_med[1]
        xyz[0,2] = xyz_med[2]
        xyz[-1,0] = xyz_med[0]
        xyz[-1,1] = xyz_med[1]
        xyz[-1,2] = xyz_med[2]

        rpy[0,0] = rpy_med[0]
        rpy[0,1] = rpy_med[1]
        rpy[0,2] = rpy_med[2]
        rpy[-1,0] = rpy_med[0]
        rpy[-1,1] = rpy_med[1]
        rpy[-1,2] = rpy_med[2]


        idx_xyz_0 = ~np.any(xyz == 0.0, axis=1)
        idx_rpy_0 = ~np.any(rpy == 0.0, axis=1) 

        return xyz, rpy, idx_xyz_0, idx_rpy_0


    def compute_statistics(self, scal, dataset):

        # Reduce with static calibration
        drpy_int = (self.xint[:, 1:4] * 180 / np.pi - np.array([scal.rx, scal.ry, scal.rz]))
        dxyz_int = (self.xint[:, 4:7] - np.array([scal.tx, scal.ty, scal.tz])) 

        # xyz min/max

        stats_str = (
            f"{dataset}, {np.min(drpy_int[:,0]):.8f}, {np.min(drpy_int[:,1]):.8f}, {np.min(drpy_int[:,2]):.8f}, "
            f"{np.min(dxyz_int[:,0]):.8f}, {np.min(dxyz_int[:,1]):.8f}, {np.min(dxyz_int[:,2]):.8f}, "
            f"{np.max(drpy_int[:,0]):.8f}, {np.max(drpy_int[:,1]):.8f}, {np.max(drpy_int[:,2]):.8f}, "
            f"{np.max(dxyz_int[:,0]):.8f}, {np.max(dxyz_int[:,1]):.8f}, {np.max(dxyz_int[:,2]):.8f}, "
            f"{np.abs(np.max(drpy_int[:,0]) - np.min(drpy_int[:,0])):.8f}, "
            f"{np.abs(np.max(drpy_int[:,1]) - np.min(drpy_int[:,1])):.8f}, "
            f"{np.abs(np.max(drpy_int[:,2]) - np.min(drpy_int[:,2])):.8f}, "
            f"{np.abs(np.max(dxyz_int[:,0]) - np.min(dxyz_int[:,0])):.8f}, "
            f"{np.abs(np.max(dxyz_int[:,1]) - np.min(dxyz_int[:,1])):.8f}, "
            f"{np.abs(np.max(dxyz_int[:,2]) - np.min(dxyz_int[:,2])):.8f}, "
            f"{np.median(drpy_int[:,0]):.8f}, {np.median(drpy_int[:,1]):.8f}, {np.median(drpy_int[:,2]):.8f}, "
            f"{np.median(dxyz_int[:,0]):.8f}, {np.median(dxyz_int[:,1]):.8f}, {np.median(dxyz_int[:,2]):.8f}, "
            f"{np.std(drpy_int[:,0]):.8f}, {np.std(drpy_int[:,1]):.8f}, {np.std(drpy_int[:,2]):.8f}, "
            f"{np.std(dxyz_int[:,0]):.8f}, {np.std(dxyz_int[:,1]):.8f}, {np.std(dxyz_int[:,2]):.8f}\n"
        )

        return stats_str

    def plot(self, scal):

        """
        scal: static calibration
        """

        fontsize_label = 50
        fontsize_xticks = 50
        
        # Change in the calibration with respect to static calibration
        drpy = (self.x[:, 1:4]* 180 / np.pi - np.array([scal.rx, scal.ry, scal.rz]))
        dxyz = (self.x[:, 4:7] - np.array([scal.tx, scal.ty, scal.tz])) * 1000

        drpy_int = (self.xint[:, 1:4] * 180 / np.pi - np.array([scal.rx, scal.ry, scal.rz]))
        dxyz_int = (self.xint[:, 4:7] - np.array([scal.tx, scal.ty, scal.tz])) * 1000

        t0 = self.x[0,0]
        t1 = self.x[-1,0] - t0
        
        # Create subplots
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot positions (xyz)
        axs[0].plot(self.x[:,0]-t0, dxyz[:, 0], "xr")   
        axs[0].plot(self.xint[:,0]-t0, dxyz_int[:, 0], color='r', linestyle='-', label='$\Delta$x')

        axs[0].plot(self.x[:,0]-t0, dxyz[:, 1], "xg")
        axs[0].plot(self.xint[:,0]-t0, dxyz_int[:, 1], color='g', linestyle='-', label='$\Delta$y')

        axs[0].plot(self.x[:,0]-t0, dxyz[:, 2], "xb")
        axs[0].plot(self.xint[:,0]-t0, dxyz_int[:, 2], color='b', linestyle='-', label='$\Delta$z')

        #axs[0].set_title(f'Translation', fontsize=fontsize_label)
        axs[0].set_ylabel('$\Delta \mathbf{t}$ (mm)', fontsize=fontsize_label, labelpad=20)
        #axs[0].legend(ncol=3, loc='lower left', fontsize=fontsize_label, bbox_to_anchor=(0.5, -0.1))
        axs[0].tick_params(axis='both', which='major', labelsize=fontsize_xticks)
        axs[0].set_xlim((0,t1))

        self.get_plt_style(axs[0])
            
        # Plot orientations (rpy)
        axs[1].plot(self.x[:,0]-t0, drpy[:, 0], "xr")
        axs[1].plot(self.xint[:,0]-t0, drpy_int[:, 0], color='r', linestyle='-', label='$\Delta$rx')

        axs[1].plot(self.x[:,0]-t0, drpy[:, 1], "xg")
        axs[1].plot(self.xint[:,0]-t0, drpy_int[:, 1], color='g', linestyle='-', label='$\Delta$ry')

        axs[1].plot(self.x[:,0]-t0, drpy[:, 2], "xb")
        axs[1].plot(self.xint[:,0]-t0, drpy_int[:, 2], color='b', linestyle='-', label='$\Delta$rz')

        #axs[1].set_title(f'Rotation', fontsize=fontsize_label)
        axs[1].set_xlabel('Time (s)', fontsize=fontsize_label)
        axs[1].set_ylabel('$\Delta \mathbf{R}$ (°)', fontsize=fontsize_label, labelpad=20)
        #axs[1].legend(loc='upper left', fontsize=13)
        #axs[1].legend(ncol=3, loc='lower left', fontsize=fontsize_label, bbox_to_anchor=(0.5, 1.1))
        axs[1].tick_params(axis='both', which='major', labelsize=fontsize_xticks)
        axs[1].set_xlim((0,t1))

        self.get_plt_style(axs[1])

        axs[0].yaxis.set_label_coords(-0.06, 0.5)  # Adjust x,y to align labels
        axs[1].yaxis.set_label_coords(-0.06, 0.5)
        
        plt.subplots_adjust(hspace=0.5)  # Increase space between plots
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()

    def get_plt_style(self, ax):
        ax.grid(which='both', axis='both', linestyle='--', linewidth=0.5)
        ax.minorticks_on()
        ax.grid(which='minor', axis='both', linestyle=':', linewidth=0.5)
        font = {'size': 12}
        plt.rc('font', **font)

    def read_calibration_from_file( self, pathx: str = None, path_intx: str = None ):
        
        print("--------------------------------------------------------------------------------")
        print("Reading system calibration parameters ")

        self.x = np.loadtxt( fname = pathx, delimiter = " ")

        # If interpolated calibration file is specified
        if path_intx is not None:
            self.xint = np.loadtxt( fname = path_intx, delimiter = " ")

        print("... done ")
        print("--------------------------------------------------------------------------------")    