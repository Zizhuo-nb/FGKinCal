import numpy as np 
import time 
import os
from dataclasses import dataclass
from colorama import Fore, Style

@dataclass
class frame3D:
    M: int   # ... Number of points
    XYZ: int # ... XYZ coordinates
    I: int   # ... Laser intensity
    ID: int  # ... Unique ID for each point in the frame

    def __init__( self, M = 0, XYZ = [], I = [], ID = [] ):
        self.M:   int        =  M
        self.XYZ: np.ndarray = XYZ
        self.I:   np.ndarray = I
        self.ID:  np.ndarray = ID

class LaserdataLMI:
    def __init__( self, name: str = None ):

        self.Filesize:        int           = 0  # size of the .bin file in bytes
        self.timestamps:      np.ndarray    = [] # timestamp vector of the laserprofiles
        self.frames:          frame3D       = [] # list of 3D laser frames 
        self.numberofframes:  int           = 0  # total number of frames 
        self.numberofpoints:  int           = 0  # total number of points 

        # Name Information
        self.name = name
    
    # ################################################################################################################
    # ################################################################################################################

    def readraw( self, path_to_file: str = "", mask_xyz: list = [[[0,0]],[0,0],[0,0]], steps: int = 1):
        
        """ This read function read a raw binary file of LMI data recorded with the scanner

        - path_to_file: path to the laser file
        - mask_xyz: mask for the laser points to read, [[x_min, x_max], [y_min, y_max], [z_min, z_max]], example: lmi_left: [[ 400,0], [], []], example: lmi_right: [[ -400,0], [], []]
        - steps: step size for reading, example: if steps = 10 every 10th scanline is read from file

        """
        
        # Print file information
        print(" "+30*"_")
        t = time.strftime("%H:%M:%S")
        print(( f"| {Style.BRIGHT}{Fore.GREEN}{t + ' Read LMI data from binary'}{Style.RESET_ALL}" ))
        print("| - path: ", path_to_file)
            
        # Get size of the LMI binary file
        self.Filesize = os.path.getsize( path_to_file )
        print("| - filesize: ", self.Filesize)

        # Open LMI binary file
        with open(path_to_file, 'rb') as fid:
            PosTmp = fid.tell()

            # Read Header
            Time                = np.fromfile(fid, np.double,1)
            Timestamp           = np.fromfile(fid,np.uint64,1)
            FirstFrameindex     = np.fromfile(fid,np.uint64,1)
            Stampcount          = np.fromfile(fid,np.uint32,1)
            PpcCount            = np.fromfile(fid,np.uint32,1)
            PpcExposure         = np.fromfile(fid,np.uint32,1)
            PpcZOffset          = np.fromfile(fid,np.int32,1)
            PpcXOffset          = np.fromfile(fid,np.int32,1)
            PpcZScale           = np.fromfile(fid,np.uint32,1)
            PpcXScale           = np.fromfile(fid,np.uint32,1)
            PpcWidth            = np.fromfile(fid,np.uint32,1)
            PpcArrayLenghtX     = np.fromfile(fid,np.int32,1)
            PPcDataX            = np.fromfile(fid,np.int16,PpcWidth[0])
            PpcArrayLenghtZ     = np.fromfile(fid,np.int32,1)
            PPcDataZ            = np.fromfile(fid,np.int16,PpcWidth[0])
            SysInfoCurrSpeed    = np.fromfile(fid,np.int64,1)
            SysInfoProcessDrops = np.fromfile(fid,np.int64,1)
            PiCount             = np.fromfile(fid,np.uint32,1)
            PiWidth             = np.fromfile(fid,np.uint32,1)
            PiArrayLenght       = np.fromfile(fid,np.int32,1)
            PiDataRaw           = np.fromfile(fid, np.int8,PiWidth[0])

            SizeOneFrame    = fid.tell()
            nFramesInFile   = self.Filesize / SizeOneFrame
            nFramesInFile   = np.uint32(nFramesInFile)

            fid.seek(0,0) #  seek(offset, whence) whence = 0 beginning of the file
            
            # Determine indices for frames to read
            frame_idx = np.arange( 0, nFramesInFile, steps )
            frame_idx_cnt = 0

            print("| - frame rate: ", steps)
            
            # Reserve memory for LMI object
            self.frames = [frame3D()] * (len( frame_idx ))
            self.numberofframes = len( frame_idx ) 
            self.timestamps      = np.zeros(len( frame_idx ) ) # UTC

            # number of points counter
            Np = 0
            id = 1
            
            # Loop in binary file     
            for DatPtr in range( 0, nFramesInFile ):

                # Read time information from current line
                Time = np.fromfile(fid,np.double,1)
                
                if Time.size == 0:
                    break
                
                # Read time and index of current laser frame
                Timestamp           = np.fromfile(fid,np.uint64,1)
                Frameindex          = np.fromfile(fid,np.uint64,1)
                Stampcount          = np.fromfile(fid,np.uint32,1)
                
                Frameindex          = Frameindex - FirstFrameindex + 1

                if ( np.size(Time) >= 0 ) & (Frameindex >= 0):
                    
                    # Read LMI data from binary file 
                    PpcCount                                        = np.fromfile(fid,'uint32',1)
                    PpcExposure                                     = np.fromfile(fid,'uint32',1)
                    PpcZOffset                                      = np.fromfile(fid,'int32',1)
                    PpcXOffset                                      = np.fromfile(fid,'int32',1)
                    PpcZScale                                       = np.fromfile(fid,'uint32',1)
                    PpcXScale                                       = np.fromfile(fid,'uint32',1)
                    PpcWidth                                        = np.fromfile(fid,'uint32',1)
                    PpcArrayLenghtX                                 = np.fromfile(fid,'int32',1)
                    X                                               = np.fromfile(fid,'int16',PpcWidth[0])
                    PpcArrayLenghtZ                                 = np.fromfile(fid,'int32',1)
                    Z                                               = np.fromfile(fid,'int16', PpcWidth[0])
                    CurrSpeed                                       = np.fromfile(fid,'int64',1)
                    ProcessDrops                                    = np.fromfile(fid,'int64',1)
                    PiCount                                         = np.fromfile(fid,'uint32',1)
                    PiWidth                                         = np.fromfile(fid,'uint32',1)
                    PiArrayLenght                                   = np.fromfile(fid,'int32',1)
                    I                                               = np.fromfile(fid,'uint8', PiWidth[0])

                    # Add laser data w.r.t steps to read
                    if (frame_idx[frame_idx_cnt] == DatPtr ) & (frame_idx_cnt < (len(frame_idx)-1)):
                        self.timestamps[frame_idx_cnt] = Time

                        # Just read points valid measured points
                        idx = np.where(X != -32768)
                        idx = idx[0]

                        X = X[idx]
                        Z = Z[idx]
                        I = I[idx]

                        # Add offset and scale values and calc to mm
                        PpcXScale_   = PpcXScale / 1000000
                        PpcZScale_   = PpcZScale / 1000000
                        PpcXOffset_ =  PpcXOffset / 1000
                        PpcZOffset_ =  PpcZOffset / 1000

                        # Transform laser points to sensor frame
                        X = (X * PpcXScale_) + PpcXOffset_
                        Z = (Z * PpcZScale_) + PpcZOffset_
                        Z = Z - 1150
                        
                        # ----------------------------------------------------------------------------------------------
                        # 1) Transformation from mm to m
                        # 
                        #X /= 1000
                        #Z /= 1000
                        Y = np.zeros( X.shape )
                        
                        # ----------------------------------------------------------------------------------------------
                        # 2) Mask laser data (if specified)

                        mask = np.array(mask_xyz[0])

                        # masking points
                        idx_mask = np.where(X > mask[0])
                        idx_mask = idx_mask[0]

                        X = X[idx_mask]
                        Y = Y[idx_mask]
                        Z = Z[idx_mask]
                        I = I[idx_mask]

                        idx = idx[idx_mask]

                        # ----------------------------------------------------------------------------------------------
                        # 4) Add laser frame3D to LMI data object
                        # 
                        n = len(X)  # length of the array
                        if n == 0:
                            ids = [0]
                    
                        start_id = id
                        
                        ids = list(range(start_id, start_id + n))
                        self.frames[frame_idx_cnt] = frame3D( M=n,
                                                              XYZ=np.column_stack( (X.astype(np.int32), Y.astype(np.int32), Z.astype(np.int32)) ),
                                                              ID = ids,
                                                              I=I.astype(int) )
                
                        id = start_id + n
                        # Update number of points counter
                        Np += len(Z)

                        # Update frame step counter
                        frame_idx_cnt += 1    
                    # end if laser frame steps
                # end if time length and frame index size
            # end reading loop
            self.numberofpoints = Np
            
        # Close LMI binary file
        fid.close()

        idxout = np.where( self.timestamps == 0.0 )[0]
        self.numberofframes -= len(idxout)
        self.timestamps = np.delete(self.timestamps, idxout)

        for j in range(0,len(idxout)):
            del self.frames[idxout[j]]
        
        # Print LMI laser data reading report
        print("| - #frames: ", len(self.frames))
        print("| - #points:", Np)
        print("| - time interval:", self.timestamps[0], self.timestamps[-1])
        print("| ... done ")
        print("| "+30*"_")

    # ################################################################################################################
    # ################################################################################################################

    def readbin( self, path_to_file):
        """ This read function read the binary file of laser data wrote with write function

        - path_to_file: path to the laser file

        """

        t = time.strftime("%H:%M:%S")
        print(( f"| {Style.BRIGHT}{Fore.GREEN}{t + ' Read LMI data from folder'}{Style.RESET_ALL}" ))
        print("| - path: ", path_to_file)

        self.Filesize:        int           = 0
        self.timestamps:      np.ndarray    = []
        self.frames:          frame3D       = []
        self.numberofframes:  int           = 0
        self.numberofpoints:  int           = 0

        with open(path_to_file, 'rb') as fid:

            self.Filesize = os.path.getsize(path_to_file)
            id = 1
            while True:
                # Read Time (timestamp)
                Time = np.fromfile(fid, dtype=np.double, count=1)
                if Time.size == 0:
                    break
                self.timestamps.append(Time[0])
                
                # Read number of points
                num_points = np.fromfile(fid, dtype=np.uint32, count=1)
                if num_points.size == 0:
                    break
                num_points = num_points[0]
                
                # Read XYZ data
                X = np.fromfile(fid, dtype=np.float32, count=num_points)
                Y = np.fromfile(fid, dtype=np.float32, count=num_points)
                Z = np.fromfile(fid, dtype=np.float32, count=num_points)
                XYZ = np.column_stack((X, Y, Z))
                
                # Read Intensity data
                I = np.fromfile(fid, dtype=np.uint32, count=num_points)
                
                n = len(X)  # length of the array
                if n == 0:
                    ids = [0]
            
                start_id = id
                        
                ids = list(range(start_id, start_id + n))
                
                # Create frame3D object and append to frames list
                self.frames.append(frame3D(M=len(X), XYZ=XYZ, ID = ids, I=I))
                id = start_id + n
                self.numberofpoints += len(X)

        self.numberofframes = len(self.timestamps)

        print("| - number of profiles: ", self.numberofframes)
        print("| - number of points: ", self.numberofpoints)
        print("| - timestamps: ", self.timestamps[0], " - ", self.timestamps[-1])
        print("|")

    # ################################################################################################################
    # ################################################################################################################

    def write_to_binary_file( self, path_to_file ):

        """ This function writes a raw binary file of LMI data to a .bin file

        - path_to_file: path to the laser file

        """

        # Print file information
        print(" "+30*"_")
        t = time.strftime("%H:%M:%S")
        print(( f"| {Style.BRIGHT}{Fore.GREEN}{t + ' Write LMI data to binary'}{Style.RESET_ALL}" ))
        print("| - path: ", " Write LMI data to .bin file")
        print("| - #points: ", self.numberofpoints)
        print("| - #frames: ", self.numberofframes)

        with open(path_to_file, 'wb') as fid:
            for i in np.arange( 0, len(self.timestamps)-1 ):

                # Write Time (timestamp)
                fid.write(np.array([self.timestamps[i]], dtype=np.double))
                        
                # Write number of points
                num_points = len(self.frames[i].I)
                    
                # Write number 
                fid.write(np.array([num_points], dtype=np.uint32))
                        
                # Write XYZ data
                    
                fid.write(self.frames[i].XYZ[:, 0].astype(np.float32))  # X data
                fid.write(self.frames[i].XYZ[:, 1].astype(np.float32))  # Y data
                fid.write(self.frames[i].XYZ[:, 2].astype(np.float32))  # Z data
                        
                # Write Intensity data
                fid.write(self.frames[i].I.astype(np.uint32))  # Intensity data

        print("| - ... done ")
        print("|")

    # ################################################################################################################
    # ################################################################################################################

    def intersecting( self, timeE: np.ndarray, return_new_object: bool = False ):

        """ This function intersects the laserdata with an timestamp vector and returns the laserdata that fall into the interval

        - timeE: vector that contains the timesstamps of the source
        - return_new_object: specifies if the class object itself should be intersected or a class objects should be returned

        """

        # Determine laser data inside the input time vector
        idx_out1 = np.where( ( self.timestamps <= timeE[0] ) ) # index outside in beginning
        idx_out2 = np.where( ( self.timestamps >= timeE[-1] ) ) # index outside in the end
        idx_out = np.r_[ (idx_out1[0], idx_out2[0]) ] # fused index

        # Count skipped scan points in the beginning
        nppoints = 0

        # Count points that are out of the interval
        if idx_out1[0].size != 0:
            for indices in idx_out1:
                i = indices[0]
                nppoints += len(self.frames[i].ID)            

        # if return new laser object
        if return_new_object == False:
            self.timestamps      = np.delete(self.timestamps, idx_out )
            self.frames          = np.delete(self.frames, idx_out)
            self.numberofframes  = self.numberofframes - len(idx_out)
            
            # restimate the number of total points
            p_cnt = 0
            for idx in range( 0, len(self.frames) ):
                p_cnt += self.frames[idx].M
                self.frames[idx].ID = [id_value - (nppoints+1) for id_value in self.frames[idx].ID]
            
            self.numberofpoints = p_cnt
        
        elif return_new_object == True:

            laserdata_ = LaserdataLMI()
            laserdata_.timestamps = np.delete(self.timestamps, idx_out )
            laserdata_.frames = np.delete(self.frames, idx_out)
            laserdata_.numberofframes  = self.numberofframes - len(idx_out)

            # restimate the number of total points
            p_cnt = 0
            for idx in range( 0, len(laserdata_.frames)-1 ):
                p_cnt += laserdata_.frames[idx].M
            
            laserdata_.numberofpoints = p_cnt

            return laserdata_ 
    
    # ################################################################################################################
    # ################################################################################################################

    def crop_by_index(self, idx):

        laserdata_ = LaserdataLMI()
        laserdata_.timestamps     = self.timestamps[ idx ]
        laserdata_.frames         = self.frames[ idx ]
        laserdata_.numberofframes = len( idx )
            
        # restimate the number of total points
        p_cnt = 0
        for i in range( 0, len(laserdata_.frames)):
            p_cnt += laserdata_.frames[i].M
            
        laserdata_.numberofpoints = p_cnt
        laserdata_.numberofframes = len(laserdata_.frames)

        return laserdata_
