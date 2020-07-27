'''
This class can be used to feed input from an image, webcam, or video to your model.
Sample usage:
    feed=InputFeeder(input_type='video', input_file='video.mp4')
    feed.load_data()
    for batch in feed.next_batch():
        do_something(batch)
    feed.close()
'''
import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys
import itertools 
import cv2
from numpy import ndarray

class InputFeeder:
    def __init__(self, input_type, input_file=None):
        '''
        input_type: str, The type of input. Can be 'video' for video file, 'image' for image file,
                    or 'cam' to use webcam feed.
        input_file: str, The file that contains the input image or video file. Leave empty for cam input_type.
        '''
        self.input_type=input_type
        if input_type=='video' or input_type=='image':
            self.input_file=input_file
    
    def load_data(self):
        if self.input_type=='video':
            self.cap=cv2.VideoCapture(self.input_file)
        elif self.input_type=='cam':
            self.cap=cv2.VideoCapture(0)
        else:
            self.cap=cv2.imread(self.input_file)

    def next_batch(self):
        '''
        Returns the next image from either a video file or webcam.
        If input_type is 'image', then it returns the same image.
        '''
        
        
        while True:
            for _ in range(10):
                ret, frame=self.cap.read()
                
            yield ret,frame


    def close(self):
        '''
        Closes the VideoCapture.
        '''
        if not self.input_type=='image':
            self.cap.release()

'''def main(args):
     
     
     pd= InputFeeder("video", args.video)
     pd.load_data()
     frame_count=0
     for  ret,frame in pd.next_batch():
        if not ret:
            break
        frame_count+=1
        cv2.imshow("frame",cv2.resize(frame,(500,500)))
        cv2.waitKey(1)
     
if __name__=='__main__':
    parser=argparse.ArgumentParser()
   
    parser.add_argument('--video', default=None)
    
    args=parser.parse_args()

    
    
main(args)'''
