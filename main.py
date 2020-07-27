
import argparse
import os
import sys
import time
import socket
import json
import cv2
import numpy as np
from random import randint
from argparse import ArgumentParser
import logging
from facial_landmarks_detection import FacialLandmarkDetection
from gaze_estimation import GazeEstimation
from face_detection import FaceDetection
from head_pose_estimation import HeadPoseEstimation
from mouse_controller import MouseController
from input_feeder import InputFeeder
import math




def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-f", "--facedetectionmodel", required=True, type=str,
                        help="Path to an xml file with a trained model for facial detection.")
    parser.add_argument("-hp", "--headposemodel", required=True, type=str,
                        help="Path to an xml file with a trained model for headpose estimation.")
    parser.add_argument("-fl", "--faciallandmarkmodel", required=True, type=str,
                        help="Path to an xml file with a trained model for facial landmark detection.")
    parser.add_argument("-g", "--gazeestimationnmodel", required=True, type=str,
                        help="Path to an xml file with a trained model for gazeestimation.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file or webcam feed,use'0' for webcam")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-flags","--previewflags",required=False,nargs='+',default=[],
                        help="To view the outputs from dfifferent models specify the flags fd,fld,hp,ge like --flags fd fld hp (seperated by space)"
                        "fd for face detection , fld for facial landmarks detection , hp for head pose estimation and ge for gaze estimation")
    
    return parser

    

def main():

    # Grab command line args
   
    logger = logging.getLogger()
    args = build_argparser().parse_args()
    flags=args.previewflags
    inputfile=args.input
    inputfeed=None
    if inputfile.lower()=='cam':
        inputfeed=InputFeeder('cam')
    elif inputfile.endswith('.jpg') or inputfile.endswith('.bmp') :
        inputfeed=InputFeeder("image",inputfile)
    #elif inputfile.endswith('.mp4') :
        
        #inputfeed=InputFeeder("video",inputfile)
    else:
        
        if not (os.path.isfile(inputfile)):
            print((inputfile))
            logger.error("Specified input file doesn't exist")
            exit(1)
        inputfeed=InputFeeder("video",inputfile)
      
    model_paths={'GazeEstimation':args.gazeestimationnmodel,'FacialLandmarkDetection':args.faciallandmarkmodel,'HeadPoseEstimation':args.headposemodel,'FaceDetection':args.facedetectionmodel}

    for file in model_paths.keys():
        
        if not os.path.isfile(model_paths[file]):
            logger.error("Unable to find specified "+file+" xml file")
        
    flm=FacialLandmarkDetection(model_paths['FacialLandmarkDetection'],args.device,args.cpu_extension)
        
    gze= GazeEstimation(model_paths['GazeEstimation'],args.device,args.cpu_extension)
        
    hpe=HeadPoseEstimation(model_paths['HeadPoseEstimation'],args.device,args.cpu_extension)
        
    fd=FaceDetection(model_paths['FaceDetection'],args.device,args.cpu_extension)
     
    flm.load_model() 
    fd.load_model()
    gze.load_model()
    hpe.load_model()
    mc=MouseController('medium','fast')
    inputfeed.load_data()
    
    frame_count=0
    for ret, frame in inputfeed.next_batch():
          
          if not ret:
            break
          frame_count+=1
          
          if frame_count%3==0:
                cv2.imshow('video',cv2.resize(frame,(300,300)))
                cv2.waitKey(1)
          facecoords,cropped_image=fd.predict(frame.copy(),args.prob_threshold)
          
          if type(cropped_image)==int:
              logger.error('unable to detect face')
          head_out=hpe.predict(cropped_image)
          left_eye,right_eye,eye=flm.predict(cropped_image)
          mouse_coords,gaze_vector=gze.predict(left_eye,right_eye,head_out)
          if (len(flags)!=0):
              preview_frame=frame.copy()
              if 'fd' in flags:
                  preview_frame=cropped_image
              if 'fld' in flags:
                  cv2.rectangle(cropped_image,(eye[0][0]-15,eye[0][1]-15),(eye[0][2]+15,eye[0][3]+15),(0,0,255))
                  cv2.rectangle(cropped_image,(eye[1][0]-15,eye[1][1]-15),(eye[1][2]+15,eye[1][3]+15),(0,0,255))
              if 'hp' in flags:
                  
                  cv2.putText(preview_frame,"Pose Angles: roll:{:2f}|pitch:{:2f}|yaw:{:2f}|".format(head_out[2],head_out[1],head_out[0]),(10,20),cv2.FONT_HERSHEY_COMPLEX,0.25,(255,0,0),1)
              if 'ge' in flags:
                  x,y,w=(int(gaze_vector[1]*12),int(gaze_vector[0]*12),130)
                  left =cv2.line(left_eye.copy(), (x-w, y-w), (x+w, y+w), (255,0,0), 2)
                  cv2.line(left, (x-w, y+w), (x+w, y-w), (255,0,0), 2)
                  right = cv2.line(right_eye.copy(), (x-w, y-w), (x+w, y+w), (255,0,0), 2)
                  cv2.line(right, (x-w, y+w), (x+w, y-w), (255,0,0), 2)
                  cropped_image[eye[0][1]:eye[0][3],eye[0][0]:eye[0][2]] = left
                  cropped_image[eye[1][1]:eye[1][3],eye[1][0]:eye[1][2]] = right
              cv2.imshow("visualisation_frame",cv2.resize(preview_frame,(300,300)))
          if frame_count%3==0:
              mc.move(mouse_coords[0],mouse_coords[1])
    logger.error("video ended...")
    cv2.destroyAllWindows()
    inputfeed.close()    
                
 

if __name__ == '__main__':
    main()
