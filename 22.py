# Copyright (C) 2019 Eugene a.k.a. Realizator, stereopi.com, virt2real team
#
# This file is part of StereoPi tutorial scripts.
#
# StereoPi tutorial is free software: you can redistribute it 
# and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the 
# License, or (at your option) any later version.
#
# StereoPi tutorial is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with StereoPi tutorial.  
# If not, see <http://www.gnu.org/licenses/>.
#
#          <><><> SPECIAL THANKS: <><><>
#
# Thanks to Adrian and http://pyimagesearch.com, as a lot of
# code in this tutorial was taken from his lessons.
#  
# Thanks to RPi-tankbot project: https://github.com/Kheiden/RPi-tankbot
#
# Thanks to rakali project: https://github.com/sthysel/rakali

#We have a pointcloud, but we lost a disparity...

from picamera import PiCamera
import time
import cv2
import numpy as np
import json
from datetime import datetime

print ("You can press 'Q' to quit this script!")
time.sleep (5)

# Visualization settings
showDisparity = True
showUndistortedImages = True
showColorizedDistanceLine = True

# Depth map default preset
SWS = 5
PFS = 5
PFC = 29
MDS = -30
NOD = 160
TTH = 100
UR = 10
SR = 14
SPWS = 100

# Camera settimgs
cam_width = 1280
cam_height = 480

# Final image capture settings
scale_ratio = 0.5

# Camera resolution height must be dividable by 16, and width by 32
cam_width = int((cam_width+31)/32)*32
cam_height = int((cam_height+15)/16)*16
print ("Used camera resolution: "+str(cam_width)+" x "+str(cam_height))

# Buffer for captured image settings
img_width = int (cam_width * scale_ratio)
img_height = int (cam_height * scale_ratio)
capture = np.zeros((img_height, img_width, 4), dtype=np.uint8)
print ("Scaled image resolution: "+str(img_width)+" x "+str(img_height))

# Depth Map colors autotune
autotune_min = 10000000
autotune_max = -10000000

# Initialize the camera
camera = PiCamera(stereo_mode='side-by-side',stereo_decimate=False)
camera.resolution=(cam_width, cam_height)
camera.framerate = 20
#camera.hflip = True

# Initialize interface windows
cv2.namedWindow("Image")
cv2.moveWindow("Image", 50,100)
cv2.namedWindow("left")
cv2.moveWindow("left", 450,100)
cv2.namedWindow("right")
cv2.moveWindow("right", 850,100)


disparity = np.zeros((img_width, img_height), np.uint8)
sbm = cv2.StereoBM_create(numDisparities=0, blockSize=21)

def stereo_depth_map(rectified_pair):
    dmLeft = rectified_pair[0]
    dmRight = rectified_pair[1]
    disparity = sbm.compute(dmLeft, dmRight)
    local_max = disparity.max()
    local_min = disparity.min()
    #print(local_max, local_min)
    # "Jumping colors" protection for depth map visualization
    global autotune_max, autotune_min
    autotune_max = max(autotune_max, disparity.max())
    autotune_min = min(autotune_min, disparity.min())

    disparity_grayscale = (disparity-autotune_min)*(65535.0/(autotune_max-autotune_min))
    disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0/65535.0))
    disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)
    return disparity_color, disparity_fixtype, disparity.astype(np.float32) / 16.0

def load_map_settings( fName ):
    global SWS, PFS, PFC, MDS, NOD, TTH, UR, SR, SPWS, loading_settings
    print('Loading parameters from file...')
    f=open(fName, 'r')
    data = json.load(f)
    SWS=data['SADWindowSize']
    PFS=data['preFilterSize']
    PFC=data['preFilterCap']
    MDS=data['minDisparity']
    NOD=data['numberOfDisparities']
    TTH=data['textureThreshold']
    UR=data['uniquenessRatio']
    SR=data['speckleRange']
    SPWS=data['speckleWindowSize']    
    #sbm.setSADWindowSize(SWS)
    sbm.setPreFilterType(1)
    sbm.setPreFilterSize(PFS)
    sbm.setPreFilterCap(PFC)
    sbm.setMinDisparity(MDS)
    sbm.setNumDisparities(NOD)
    sbm.setTextureThreshold(TTH)
    sbm.setUniquenessRatio(UR)
    sbm.setSpeckleRange(SR)
    sbm.setSpeckleWindowSize(SPWS)
    f.close()
    print ('Depth map settings are loaded from the file '+fName)

# Loading depth map settings
load_map_settings ("3dmap_set.txt")

# Loading stereoscopic calibration data
try:
    npzfile = np.load('./calibration_data/{}p/stereo_camera_calibration.npz'.format(img_height))
except:
    print("Camera calibration data not found in cache, file ", './calibration_data/{}p/stereo_camera_calibration.npz'.format(img_height))
    exit(0)
try:
    npzright = np.load('./calibration_data/{}p/camera_calibration_right.npz'.format(img_height))
except:
    print("Camera calibration data not found in cache, file ", './calibration_data/{}p/camera_calibration_right.npz'.format(img_height))
    exit(0)    
    
    
imageSize = tuple(npzfile['imageSize'])
leftMapX = npzfile['leftMapX']
leftMapY = npzfile['leftMapY']
rightMapX = npzfile['rightMapX']
rightMapY = npzfile['rightMapY']
QQ = npzfile['dispartityToDepthMap']
right_K = npzright['camera_matrix']
#print (right_K)
#print (QQ)
#exit(0)

map_width = 320
map_height = 240

min_y = 10000
max_y = -10000
min_x =  10000
max_x = -10000
min_z =  10000
max_z = -10000

def remove_invalid(disp_arr, points, colors ):
    mask = (
        (disp_arr > disp_arr.min()) &
        #(disp_arr < disp_arr.max()) &
        np.all(~np.isnan(points), axis=1) &
        np.all(~np.isinf(points), axis=1) 
    )
    #print ("Points shape: "+str(points.shape))
    #print ("Points masked shape: "+str(points[mask].shape))
    #print ("Mask shape: "+str(mask.shape))
    #print ("Colors shape: "+str(colors.shape))
    #zuzu = colors.reshape(-1,3)
    #print ("Colors REshape: "+str(zuzu.shape))
    
    return points[mask], colors[mask]
    #return points, colors
def calc_point_cloud(image, disp, q):
    points = cv2.reprojectImageTo3D(disp, q).reshape(-1, 3)
    #points[:,2] -= 0.3
    #points[:,2] *= 5.0
    #miny = np.amin(points[:,1])
    #maxy = np.amax(points[:,1])
    minz = np.amin(points[:,2])
    maxz = np.amax(points[:,2])
    print("Min Z: " + str(minz))
    print("Max Z: " + str(maxz))
    #print("Min Y: " + str(miny))
    #print("Max Y: " + str(maxy))
    
    #if our image is color or black and white?
    image_dim = image.ndim
    #print ("Image shape: " + str(image.shape))
    #print ("Image dim: "+ str(image_dim))
    if (image_dim == 2):  # grayscale
        colors = image.reshape(-1, 1)
    elif (image_dim == 3): #color
        colors = image.reshape(-1, 3)
    else:
        print ("Wrong image data")
        exit (0)
    #colors = image.reshape(-1, 3)
    return remove_invalid(disp.reshape(-1), points, colors)

def calc_projected_image(points, colors, r, t, k, dist_coeff, width, height):
    xy, cm = project_points(points, colors, r, t, k, dist_coeff, width, height)
    image = np.zeros((height, width, 3), dtype=colors.dtype)
    image[xy[:, 1], xy[:, 0]] = cm
    return image

def project_points(points, colors, r, t, k, dist_coeff, width, height):
    projected, _ = cv2.projectPoints(points, r, t, k, dist_coeff)
    xy = projected.reshape(-1, 2).astype(np.int)
    mask = (
        (0 <= xy[:, 0]) & (xy[:, 0] < width) &
        (0 <= xy[:, 1]) & (xy[:, 1] < height)
    )
    #print ("Colors shape: "+str(colors.shape))
    #colors.reshape(-1,3)
    #print ("Colors REshape: "+str(colors.shape))
    colorsreturn = colors[mask]
    return xy[mask], colorsreturn

def rotate(arr, anglex, anglez):
    return np.array([  # rx
        [1, 0, 0],
        [0, np.cos(anglex), -np.sin(anglex)],
        [0, np.sin(anglex), np.cos(anglex)]
    ]).dot(np.array([  # rz
        [np.cos(anglez), 0, np.sin(anglez)],
        [0, 1, 0],
        [-np.sin(anglez), 0, np.cos(anglez)]
    ])).dot(arr)

angles = {  # x, z
    'w': (-np.pi/4, 0),
    's': (np.pi/4, 0),
    'a': (0, np.pi/4),
    'd': (0, -np.pi/4)
    }
r = np.eye(3)
t = np.array([0, 0.0, 100.5])

# Capture the frames from the camera
for frame in camera.capture_continuous(capture, format="bgra", use_video_port=True, resize=(img_width,img_height)):
    t1 = datetime.now()
    pair_img = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY)
    
    # Cutting stereopair to the left and right images
    imgLeft = pair_img [0:img_height,0:int(img_width/2)] #Y+H and X+W
    imgRight = pair_img [0:img_height,int(img_width/2):img_width] #Y+H and X+W
    
    # Undistorting images
    imgL = cv2.remap(imgLeft, leftMapX, leftMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    imgR = cv2.remap(imgRight, rightMapX, rightMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    #imgRcut = imgR
    #imgLcut = imgL
    #rectified_pair = (imgLcut, imgRcut)
    imgRcut = imgR [120:200,0:int(img_width/2)]
    imgLcut = imgL [120:200,0:int(img_width/2)]
    rectified_pair = (imgLcut, imgRcut)
    
    # Disparity map calculation
    disparity, disparity_bw, native_disparity  = stereo_depth_map(rectified_pair)

    # Point cloud calculation   
    points_3, colors = calc_point_cloud(disparity, native_disparity, QQ)

    # Camera settings for the pointcloud projection 
    k = right_K
    dist_coeff = np.zeros((4, 1))

    
    if (showUndistortedImages):
        cv2.imshow("left", imgLcut)
        cv2.imshow("right", imgRcut)    
    t2 = datetime.now()
    if (showDisparity):
        cv2.imshow("Image", disparity)
    key = cv2.waitKey(1) & 0xFF   
    if key == ord("q"):
        quit();
    ch = chr(key)
    if ch in angles:
        ax, az = angles[ch]
        r = rotate(r, -ax, -az)
        #t = rotate(t, ax, az)
    if ch == "1":   # decrease camera distance from the point cloud
        t[2] -= 100
    elif ch == "2": # decrease camera distance from the point cloud
        t[2] += 100
    
    projected_image = calc_projected_image(points_3, colors, r, t, k, dist_coeff, map_width, map_height)
    cv2.imshow("XY projection", projected_image)     
#print ("DM build time: " + str(t2-t1))


