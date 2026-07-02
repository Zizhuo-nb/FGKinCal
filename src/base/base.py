import numpy as np

# Find duplicates indices in vector
def find_duplicate( vector ):
    _, unique_indices, counts = np.unique( vector, return_index=True, return_counts=True)
    duplicate_indices = unique_indices[counts > 1]
    return duplicate_indices

# Rotation Matrix X from Angle Xr
def RotmatX( alpha ):
    return np.array([ [1,0,0] , [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)] ])

# Rotation Matrix Y from Angle Yr
def RotmatY( beta ):
    return np.array([ [np.cos(beta), 0, np.sin(beta) ], [0,1,0], [-np.sin(beta), 0, np.cos(beta)] ])

# Rotation Matrix Z from Angle Zr
def RotmatZ( gamma ):
    return np.array([ [np.cos(gamma),-np.sin(gamma), 0] , [np.sin(gamma), np.cos(gamma), 0] , [0,0,1] ])

# Creates a homogeneous transformation matrix from input rotation and translation
def create_homogeneous_matrix(R, t):
    if R.shape != (3, 3):
        raise ValueError("Rotation matrix R must be 3x3.")
        
    if t.shape != (3,) and t.shape != (3, 1):
        raise ValueError("Translation vector t must be a 3x1 vector.")
        
    t = t.reshape(3, 1)   
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t.flatten()  
    return H

# Extracts euler angles from a rotation matrix
def Rotmat2Euler( rotmat ):

    rX  = np.arctan2( rotmat[2,1], rotmat[2,2] )
    rY = np.arctan2( -rotmat[2,0], np.sqrt(rotmat[2,1]**2 + rotmat[2,2]**2) )
    rZ   = np.arctan2( rotmat[1,0], rotmat[0,0] )
    rXrYrZ = np.array( [rX, rY, rZ] )
    return rXrYrZ

# Construcs a rotation matrix from input Euler angles
def Euler2RotMat(roll, pitch, yaw):
    R = RotmatZ( yaw ) @ RotmatY( pitch ) @ RotmatX( roll )
    return R