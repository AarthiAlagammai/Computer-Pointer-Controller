'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys
import math
import itertools 
#l=r'C:\Aarthi\Openvino\openvino\deployment_tools\inference_engine\bin\intel64\Release\cpu_extension_avx2.dll'#'/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so'
class GazeEstimation:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_weights=model_name.split(".")[0]+'.bin'
        self.model_structure=model_name
        self.device=device
        self.extensions=extensions
        #raise NotImplementedError

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.plugin = IECore()
        self.network = IENetwork(model=self.model_structure, weights=self.model_weights)
        supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        
        
        if len(unsupported_layers)!=0 and self.device=='CPU':
            print("unsupported layers found:{}")#.format(unsupported_layers))
            if not self.extensions==None:
                print("Adding cpu_extension")
                self.plugin.add_extension(self.extensions, self.device)
                supported_layers = self.plugin.query_network(network = self.network, device_name=self.device)
                unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
                if len(unsupported_layers)!=0:
                    print("After adding the extension still unsupported layers found")
                    exit(1)
                print("After adding the extension the issue is resolved")
            else:
                print("Give the path of cpu extension")
                exit(1)
                
        self.exec_net = self.plugin.load_network(network=self.network, device_name=self.device,num_requests=1)
        self.input_name = [i for i in self.network.inputs.keys()]
        #print(self.input_name)
        self.input_shape=self.network.inputs[self.input_name[1]].shape
        #print(self.input_shape)
        self.output_name=[i for i in self.network.outputs.keys()]
        #print(self.output_name)
        
        #raise NotImplementedError

    def predict(self, left_image,right_image,pose_vec):
        
        self.processed_left,self.processed_right=self.preprocess_input(left_image,right_image)
        input_dict={"head_pose_angles": pose_vec,"left_eye_image":self.processed_left,
                "right_eye_image":self.processed_right}
        res=self.exec_net.infer(input_dict)
        #print(res)
        detection=res[self.output_name[0]]
        mouse_coord,gaze_vector=self.preprocess_output(detection,pose_vec)
        return mouse_coord,gaze_vector
        
        #raise NotImplementedError

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, left_image,right_image):
    
        left_eye=cv2.resize(left_image,(self.input_shape[3],self.input_shape[2]))
        left_eye=left_eye.transpose((2,0,1))
        left_eye=left_eye.reshape(1,*left_eye.shape)
        right_eye=cv2.resize(right_image,(self.input_shape[3],self.input_shape[2]))
        right_eye=right_eye.transpose((2,0,1))
        right_eye=right_eye.reshape(1,*left_eye.shape)
        return left_eye,right_eye
        
        #raise NotImplementedError

    def preprocess_output(self, outputs,pose_vec):
        gaze_vector=outputs[0].tolist()
        angle_r_fc=pose_vec[2]
        cosine=math.cos(angle_r_fc*math.pi/180.0)
        sine=math.sin(angle_r_fc*math.pi/180.0)
        x_coord=gaze_vector[0]*cosine+gaze_vector[1]*sine
        y_coord=-gaze_vector[0]*sine+gaze_vector[1]*cosine
        return (x_coord,y_coord),gaze_vector
        #raise NotImplementedError


