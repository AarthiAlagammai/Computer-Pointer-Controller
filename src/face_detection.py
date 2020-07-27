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
class FaceDetection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        self.model_weights=model_name.split(".")[0]+'.bin'
        self.model_structure=model_name
        self.device=device
        self.extensions=extensions
        self.device=device
        
    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.core=IECore()      

        if self.extensions and "CPU" in self.device:
            self.core.add_extension(self.extensions, self.device)
        self.network=IENetwork(self.model_structure, self.model_weights)
        supported_layers=self.core.query_network(self.network,device_name=self.device)
        unsupported_layers= [l for l in self.network.layers.keys() if l in supported_layers]
        if len(unsupported_layers)!=0 and self.device=='CPU':
            print("unsupported layers found:{}")#.format(unsupported_layers))
            if not self.extensions==None:
                print("Adding cpu_extension")
                self.core.add_extension(self.extensions, self.device)
                supported_layers = self.core.query_network(network = self.network, device_name=self.device)
                unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
                if len(unsupported_layers)!=0:
                    print("After adding the extension still unsupported layers found")
                    exit(1)
                print("After adding the extension the issue is resolved")
            else:
                print("Give the path of cpu extension")
                exit(1)
        self.net=self.core.load_network(network=self.network,device_name=self.device,num_requests=1)
        self.input_name=next(iter(self.network.inputs))
        self.output_name=next(iter(self.network.outputs))
        self.input_shape=self.network.inputs[self.input_name].shape
        self.output_shape=self.network.outputs[self.output_name].shape
        
        #raise NotImplementedError
        

    def predict(self, image,prob_threshold):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        width=image.shape[1]
        
        height=image.shape[0]
        self.processed_image=self.preprocess_input(image)
        
        input_dict={self.input_name:self.processed_image}
        res=self.net.infer(input_dict)
        detection=res[self.output_name]
        coords=self.preprocess_output(detection,prob_threshold,width,height)
        if (len(coords)==0):
            return 0,0
        coords_face=coords[0]
        cropped_face = image[coords_face[1]:coords_face[3], coords_face[0]:coords_face[2]]
        return coords,cropped_face
        #raise NotImplementedError

    #def check_model(self):
        #raise NotImplementedError

    def preprocess_input(self, image):
    
        #image_processed = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_processed=cv2.resize(image,(self.input_shape[3],self.input_shape[2]))
        image_processed=image_processed.transpose((2,0,1))
        image_processed=image_processed.reshape(1,*image_processed.shape)
        return image_processed
        #raise NotImplementedError

    def preprocess_output(self, outputs,prob_threshold,width,height):
    
        coord=[]
        for box in outputs[0][0]:
            if box[2]>prob_threshold:
                xmin = int(box[3] * width)
                ymin = int(box[4] * height)
                xmax = int(box[5] * width)
                ymax = int(box[6] * height)
                coord.append((xmin, ymin, xmax, ymax))
        return coord
        #raise NotImplementedError
