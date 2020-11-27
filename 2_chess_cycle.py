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

import os
import time
from datetime import datetime
import picamera
from picamera import PiCamera
import cv2
import numpy as np

# User quit method message 
print("You can press 'Q' to quit this script.")

# Photo session settings
total_photos = 50             # Number of images to take
countdown = 10                 # Interval for count-down timer, seconds
font=cv2.FONT_HERSHEY_SIMPLEX # Cowntdown timer font
 
# Camera settimgs
cam_width = 1280              # Cam sensor width settings
cam_height = 480              # Cam sensor height settings

# Final image capture settings
scale_ratio = 1

# Camera resolution height must be dividable by 16, and width by 32
cam_width = int((cam_width+31)/32)*32
cam_height = int((cam_height+15)/16)*16
print ("Camera resolution: "+str(cam_width)+" x "+str(cam_height))

# Buffer for captured image settings
img_width = int (cam_width * scale_ratio)
img_height = int (cam_height * scale_ratio)
capture = np.zeros((img_height, img_width, 4), dtype=np.uint8)
print ("Scaled image resolution: "+str(img_width)+" x "+str(img_height))

# Initialize the camera
camera = PiCamera(stereo_mode='side-by-side', stereo_decimate=False)
camera.resolution=(cam_width, cam_height)
camera.framerate = 20
camera.hflip = True

# Lets start taking photos! 
counter = 0
t2 = datetime.now()
print ("Starting photo sequence")
for frame in camera.capture_continuous(capture, format="bgra", \
                  use_video_port=True, resize=(img_width,img_height)):
    t1 = datetime.now()
    cntdwn_timer = countdown - int ((t1-t2).total_seconds())
    
    # If cowntdown is zero - let's record next image
    if cntdwn_timer == -1:
      counter += 1
      filename = './scenes/scene_'+str(img_width)+'x'+str(img_height)+'_'+\
                  str(counter) + '.png'
      cv2.imwrite(filename, frame)
      print (' ['+str(counter)+' of '+str(total_photos)+'] '+filename)
      t2 = datetime.now()
      time.sleep(1)
      cntdwn_timer = 0      # To avoid "-1" timer display 
      next
    
    # Draw cowntdown counter, seconds
    cv2.putText(frame, str(cntdwn_timer), (50,50), font, 2.0, (0,0,255),4, cv2.LINE_AA)
    cv2.imshow("pair", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # Press 'Q' key to quit, or wait till all photos are taken
    if (key == ord("q")) | (counter == total_photos):
      break

 
print ("Photo sequence finished")
 
