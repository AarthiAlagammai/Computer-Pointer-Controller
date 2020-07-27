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
class HeadPoseEstimation:
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
        
        self.core=IECore()
       
        if self.extensions and "CPU" in self.device:
                self.core.add_extension(self.extensions,self.device)
        #print(self.model_weights)
        self.network=IENetwork(self.model_structure,self.model_weights)
        
        supported_layer=self.core.query_network(self.network,device_name=self.device)
        unsupported_layer= [l for l in self.network.layers.keys() if l in supported_layer]
        if len(unsupported_layer)!=0 and self.device=='CPU':
            print("unsupported layers found:{}")#.format(unsupported_layer))
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

    def predict(self, image):
        
        self.processed_image=self.preprocess_input(image)
        input_dict={self.input_name:self.processed_image}
        res=self.net.infer(input_dict)
        #print(res)
        output=self.preprocess_output(res)
        print(output)
        return output
        #detection=res[self.output_name]
        #print(detection)
        
        #raise NotImplementedError

    #def check_model(self):
        #raise NotImplementedError

    def preprocess_input(self, image):
   
        processed_image=cv2.resize(image,(self.input_shape[3],self.input_shape[2]))
        processed_image=processed_image.transpose((2,0,1))
        processed_image=processed_image.reshape(1,*processed_image.shape)
        #print(processed_image.shape)
        return processed_image
        #raise NotImplementedError

    def preprocess_output(self, outputs):
         output=[]
         #print(outputs['angle_y_fc'][0][0])
         output.append(outputs['angle_y_fc'].tolist()[0][0])
         output.append(outputs['angle_p_fc'].tolist()[0][0])
         output.append(outputs['angle_r_fc'].tolist()[0][0])
         return output
        #raise NotImplementedError
'''def main(args):
     model='/home/workspace/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml'
     device=args.device
     image='/home/workspace/head-pose-estimation-adas-0001.png'
     
     pd= HeadPoseEstimation(model, device)
     pd.load_model()
     img=cv2.imread(image)
     pd.predict(img)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', default=None)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    
    args=parser.parse_args()

    
    
main(args)'''