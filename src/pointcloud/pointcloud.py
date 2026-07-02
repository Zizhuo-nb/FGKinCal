import numpy as np 
import time 
import os 
import laspy
import random
from colorama import init, Fore, Style
import open3d as o3d

import matplotlib.cm as cm

from plyfile import PlyData

class pointcloud:
    def __init__( self, time_frame: str = None,
                        xyz_frame: str = None,
                        no_points: int = 0,
                        glob_off: np.ndarray = np.array((0,0,0)),
                        system: str = None ):
        
        # time stamps and frame
        self.time = np.zeros( (no_points,1), dtype=float )
        self.time_frame = time_frame
        
        # global point cloud offset
        self.glob_off = glob_off
        
        # coordinates and frame
        self.xyz = np.zeros( (no_points,3), dtype=float )
        self.xyz_frame = xyz_frame
        
        # intensity
        self.intensity = np.zeros( (no_points,), dtype=float )
        self.normals = None
        self.colors = None
        self.id = None

        # Systems name the point cloud belongs to
        self.system = system
        self.id = np.zeros( (no_points,), dtype=float )

        # 2D bounding box of the point cloud
        self.bbox = [0,0,0,0]
        self.xyzc = np.array([0,0,0])
        
    def write_to_file( self, path: str = [], filename: str = [] , format: str = ".las", offset: np.ndarray = [0,0,0]) -> None:

        t = time.strftime("%H:%M:%S")
        print(( f"| {Style.BRIGHT}{Fore.GREEN}{t + ' Write point cloud to folder'}{Style.RESET_ALL}" ))
        print("| - format: ", format)
        print("| - number of points: ", len(self.time))
        print("| - path: ", path+filename+format)
        print("|")

        # Create Folder
        if not os.path.exists( path ):
            os.makedirs( path )

        if format == ".txt":
            np.savetxt(path+filename+".txt", np.c_[self.time, self.xyz, self.intensity], fmt='%.4f')

        if format == ".las":

            header = laspy.LasHeader(point_format=3, version="1.2")

            header.offsets = offset
            header.scales = np.array([0.001, 0.001, 0.001])

            # Add extra dimensions BEFORE creating LasData object
            if self.id is not None:
                header.add_extra_dim(laspy.ExtraBytesParams(name="ids", type=np.uint64))

            if self.xyz is not None:
                header.add_extra_dim(laspy.ExtraBytesParams(name="height", type=np.float32))

            # NOW create the LasData object with the modified header
            las = laspy.LasData(header)

            las.x = (self.xyz[:,0]) + offset[0]
            las.y = (self.xyz[:,1]) + offset[1]
            las.z = (self.xyz[:,2]) + offset[2]

            las.intensity = self.intensity 
            las.gps_time = np.ravel(self.time)

            # Now assign values to the extra fields
            if self.id is not None:
                las.ids = np.asarray(self.id, dtype=np.uint64).flatten()

            if self.xyz is not None:
                las.height = np.asarray(self.xyz[:,2] - self.xyz[:,2].min(), dtype=np.float32).flatten()

            las.write(path + filename + ".las")
        
        print("| ")

    def insert_points(self, time, xyz, intensity, idxA, idxB, id) -> None:
        
        #print("Inset points in point cloud object, idxA = ", idxA,  " idxB = ", idxB, "total size: ", len(self.time) )
        self.time[idxA:idxB] = time
        self.xyz[idxA:idxB] = xyz
        self.intensity[idxA:idxB] = intensity
        self.id[idxA:idxB] = id
    
    def read(self, fname):

        str_s = fname.split(".")

        # 1) LAS

        if str_s [-1] == "las":

            las = laspy.read( fname )

            # Print scalar fields of the point cloud
            #for dimension in las.point_format.dimensions:
            #    print(dimension.name)

            # Read points
            with laspy.open(fname) as las:
                for points in las.chunk_iterator(100_000_000):
                    self.xyz = np.column_stack( (points.x, points.y, points.z) )
                    self.intensity = points.intensity
                    self.time = points.gps_time
                    
            # Bounding box point cloud
            self.bbox = [np.min(self.xyz[:,0]),np.max(self.xyz[:,0]),np.min(self.xyz[:,1]),np.max(self.xyz[:,1])]
            self.xyzc = np.array([np.mean(self.xyz[:,0]), np.mean(self.xyz[:,1]), np.mean(self.xyz[:,2])])

        # 2) PLY

        if str_s [-1] == "ply":

            plydata = PlyData.read(fname)
            vertex_data = plydata['vertex'].data
            
            # xyz coordinates
            self.xyz = np.column_stack( (vertex_data['x'], vertex_data['y'], vertex_data['z']) )
            
            # time stamps
            self.time = vertex_data['scalar_GpsTime']

            # M3C2 distance
            self.m3c2_dist = vertex_data['scalar_M3C2_distance']

            # Compute bounding box and center
            self.bbox = [np.min(self.xyz[:,0]),np.max(self.xyz[:,0]),np.min(self.xyz[:,1]),np.max(self.xyz[:,1])]
            self.xyzc = np.array([np.mean(self.xyz[:,0]), np.mean(self.xyz[:,1]), np.mean(self.xyz[:,2])])


    def subsample(self, factor: int = 1, method: str = "space"):

        PC_sub = pointcloud()
        PC_sub.time_frame = self.time_frame
        PC_sub.xyz_frame = self.xyz_frame

        print( time.strftime("%H:%M:%S"), " Number of points ", self.xyz.shape )

        if method == "space":
            
            PC_sub.xyz = self.xyz[::factor,:]
            PC_sub.intensity = self.intensity[::factor]
            PC_sub.time = self.time[::factor]

            print( time.strftime("%H:%M:%S"), " Number of points after subsampling ", PC_sub.xyz.shape )

            # Compute bounding box and center
            self.bbox = [np.min(self.xyz[:,0]),np.max(self.xyz[:,0]),np.min(self.xyz[:,1]),np.max(self.xyz[:,1])]
            self.xyzc = np.array([np.mean(self.xyz[:,0]), np.mean(self.xyz[:,1]), np.mean(self.xyz[:,2])])

            return PC_sub

        elif method == "random":

            N = int( len(self.xyz) / factor )

            idx = random.sample(range(1, len(self.xyz)), N)

            PC_sub.xyz= self.xyz[idx,:]
            PC_sub.intensity = self.intensity[idx]
            PC_sub.time = self.time[idx]

            print( time.strftime("%H:%M:%S"), " Number of points after subsampling ", PC_sub.xyz.shape )

            # Compute bounding box and center
            PC_sub.bbox = [np.min(self.xyz[:,0]),np.max(self.xyz[:,0]),np.min(self.xyz[:,1]),np.max(self.xyz[:,1])]
            PC_sub.xyzc = np.array([np.mean(self.xyz[:,0]), np.mean(self.xyz[:,1]), np.mean(self.xyz[:,2])])

            return PC_sub, np.array(idx)
    
    def select_by_index( self, idx ):
        
        pc = pointcloud()
        
        pc.xyz = self.xyz[idx,:]

        if self.intensity is not None:
            pc.intensity = self.intensity[idx]
        if self.time is not None:    
            pc.time = self.time[idx]
        if self.normals is not None: 
            pc.normals = self.normals[idx,:]
        if self.colors is not None:
            pc.colors = self.colors[idx,:]

        # Compute bounding box and center
        pc.bbox = [np.min(self.xyz[:,0]),np.max(self.xyz[:,0]),np.min(self.xyz[:,1]),np.max(self.xyz[:,1])]
        pc.xyzc = np.array([np.mean(self.xyz[:,0]), np.mean(self.xyz[:,1]), np.mean(self.xyz[:,2])])
        
        return pc
    

    def concatenate( self, pc ):
        
        pcout = pointcloud()
        
        pcout.xyz = np.concatenate( (self.xyz, pc.xyz) )
        pcout.id = np.concatenate( (1*np.ones(self.time.shape), 2*np.ones(pc.time.shape)) )

        if self.intensity is not None:
            pcout.intensity = np.concatenate( (self.intensity, pc.intensity) )
        if self.time is not None:
            pcout.time = np.concatenate( (self.time, pc.time) )
        if self.normals is not None: 
            pcout.normals = np.concatenate( (self.normals, pc.normals) )
        if self.colors is not None:
            pcout.colors = np.concatenate( (self.colors, pc.colors) )

        # Compute bounding box and center
        pcout.bbox = [np.min(self.xyz[:,0]),np.max(self.xyz[:,0]),np.min(self.xyz[:,1]),np.max(self.xyz[:,1])]
        pcout.xyzc = np.array([np.mean(self.xyz[:,0]), np.mean(self.xyz[:,1]), np.mean(self.xyz[:,2])])
        
        return pcout



    def translate(self, txyz ):
        self.xyz -= txyz

    def PCAtransform(self):

        # Step 1: Calculate the mean of the point cloud
        mean = np.mean(self.xyz, axis=0)
        
        # Step 2: Center the point cloud by subtracting the mean
        centered_points = self.xyz - mean
        
        # Step 3: Calculate the covariance matrix
        covariance_matrix = np.cov(centered_points, rowvar=False)
        
        # Step 4: Perform PCA: get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Step 5: Sort eigenvalues and eigenvectors
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Transform points
        self.xyz = centered_points @ eigenvectors 


    def plot(self, colorfield = None):

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector( self.xyz )

        if colorfield is not None:
            z_min = np.min(colorfield)
            z_max = np.max(colorfield)
            z_normalized = (colorfield - z_min) / (z_max - z_min)
            colors = cm.viridis(z_normalized)[:, :3] 
            point_cloud.colors = o3d.utility.Vector3dVector(colors)

        vis = o3d.visualization.Visualizer()
        vis.create_window()
            
        vis.add_geometry( point_cloud )

        # Minimum point
        xyz_min = np.min(self.xyz, axis=0)
        xyz_max = np.max(self.xyz, axis=0)
        
        # Create coordiante origin and axes
        grid = o3d.geometry.TriangleMesh.create_coordinate_frame( size=0.5, origin = xyz_min )

        # Linien für das Gitter in der xy-Ebene
        #grid_scale_x = xyz_max[0] - xyz_min[0]
        #grid_scale_y = xyz_max[1] - xyz_min[1]
        #grid_scale_z = xyz_max[2] - xyz_min[2]

        #xy_grid_lines = o3d.geometry.LineSet()
        #xy_points = [[xyz_min[0], xyz_min[1], xyz_min[2]], [xyz_min[0] + grid_scale_x,  xyz_min[1],  xyz_min[2]], [ xyz_min[0]+grid_scale_x,  xyz_min[1] + grid_scale_y,  xyz_min[2]], [ xyz_min[0],  xyz_min[1]+grid_scale_y,  xyz_min[2]], [ xyz_min[0],  xyz_min[1],  xyz_min[2]], [ xyz_min[0],  xyz_min[1]+grid_scale_y,  xyz_min[2]]]
        #xy_lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5]]
        #xy_grid_lines.points = o3d.utility.Vector3dVector(xy_points)
        #xy_grid_lines.lines = o3d.utility.Vector2iVector(xy_lines)
        #vis.add_geometry(xy_grid_lines)
        
        # Oriented bounding box
        obb = point_cloud.get_axis_aligned_bounding_box()
        obb.color = [1, 0, 0]
        vis.add_geometry(obb)
        vis.add_geometry( grid )

        vis.run()
        vis.destroy_window()

