import time 
from src.calibration.calibration import calibration
from src.calibration.kinematiccalibration import kinematiccalibration
from src.pointcloud.pointcloud import pointcloud

from src.dataclasses.trajectory import Trajectory
from src.base.base import RotmatX, RotmatY, RotmatZ

from src.dataclasses.LMIdata import LaserdataLMI
#import src.geodetictools.transformations as geobase
import numpy as np
from alive_progress import alive_bar

class directgeoreferencing:

    trajectory: Trajectory
    laserline: LaserdataLMI
    systemcalibration: calibration
    kinematicsystemcalibration: kinematiccalibration
    pc: pointcloud

    def __init__( self, trajectory, laserlines, systemcalibration ):

        self.trajectory: Trajectory = trajectory
        self.laserlines: LaserdataLMI = laserlines
        self.systemcalibration: calibration = systemcalibration
        self.pc: pointcloud

    #######################################################################################################################################

    def run( self, calibration="static" ) -> pointcloud:
        
        if calibration == "static":
            R_BS = RotmatZ(np.deg2rad(self.systemcalibration.rz)) @ RotmatY( np.deg2rad(self.systemcalibration.ry) ) @ RotmatX( np.deg2rad(self.systemcalibration.rx) ) # body to sensor
            R_SB = R_BS.T # sensor to body
            dxyz_SB = np.array([self.systemcalibration.tx, self.systemcalibration.ty, self.systemcalibration.tz]) # translation

        # Point cloud object to store georeferenced points
        self.pc = pointcloud( time_frame = "UTC",
                              xyz_frame = "UTM",
                              no_points = self.laserlines.numberofpoints,
                              system =  self.laserlines.name)

        # Root index for point cloud insert 
        root = 0
        id = 1

        # Main georeferencing loop
        for idx in range( 0, self.trajectory.statesall.shape[0] ):

            # Kinematic system Calibration

            if calibration=="kinematic":
                R_BS = RotmatZ(self.systemcalibration.xint[idx,3]) @ RotmatY(self.systemcalibration.xint[idx,2]) @ RotmatX(self.systemcalibration.xint[idx,1]) # body to sensor
                R_SB = R_BS.T # sensor to body
                dxyz_SB = np.array([self.systemcalibration.xint[idx,4], self.systemcalibration.xint[idx,5], self.systemcalibration.xint[idx,6]]) # translation

            # Laser points in body frame
            xyz_b_right = np.dot(R_SB, (self.laserlines.frames[idx].XYZ.T) / 1000 ).T + np.tile(dxyz_SB, (len(self.laserlines.frames[idx].XYZ), 1))

            # Rotation Matrix Body to NED frame
            R_B_NED_right = np.dot(np.dot(RotmatZ(self.trajectory.statesall[idx,9]), RotmatY(self.trajectory.statesall[idx,8])), RotmatX(self.trajectory.statesall[idx,7]))

            # laser points in UTM frame
            xyz_e_right = np.dot(R_B_NED_right, xyz_b_right.T)

            # Calculate XG, YG, ZG and transform to millimeter
            XG = (xyz_e_right[0] + self.trajectory.statesall[idx,1])
            YG = (xyz_e_right[1] + self.trajectory.statesall[idx,2])
            ZG = (xyz_e_right[2] + self.trajectory.statesall[idx,3])
            id += 1
            
            # Store points in point cloud object in [mm]
            self.pc.insert_points( time = self.laserlines.timestamps[idx] * np.ones( (len(XG), 1 )),
                                   xyz = np.c_[ XG, YG, ZG ],
                                   intensity = self.laserlines.frames[idx].I,
                                   idxA = root,
                                   idxB = root + self.laserlines.frames[idx].M,
                                   id = self.laserlines.frames[idx].ID)
            
            root += self.laserlines.frames[idx].M

        # calculate 2D bounding box [mm]
        self.pc.bbox = np.array( [np.min(self.pc.xyz[:,0]), np.max(self.pc.xyz[:,0]), np.min(self.pc.xyz[:,1]), np.max(self.pc.xyz[:,1])] )
        
        return self.pc