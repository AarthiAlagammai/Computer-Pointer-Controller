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
import itertools 
class FacialLandmarkDetection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        
        self.device=device
        self.model_weights=model_name.split(".")[0]+'.bin'
        self.model_structure=model_name
        self.extensions=extensions
        self.device=device
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
        self.network=IENetwork(self.model_structure,self.model_weights)
        supported_layers=self.core.query_network(self.network,device_name=self.device)
        print(supported_layers)
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

    def predict(self, image):
        width=image.shape[1]
        height=image.shape[0]
        self.processed_image=self.preprocess_input(image)
       
        input_dict={self.input_name:self.processed_image}
        res=self.net.infer(input_dict)
        # print(res)
        detection=res[self.output_name]
        print(detection.shape)
        
        out=self.preprocess_output(detection,width,height)
        
        out = np.asarray(out)
        print(type(out))
        coords=out.astype(np.int32)
        #cv2.rectangle(image,(coords[0],coords[1]),(coords[2],coords[3]),(0,0,255))
        left_eye_xmin=coords[0]-15
        left_eye_ymin=coords[1]-15
        left_eye_xmax=coords[0]+15
        left_eye_ymax=coords[1]+15
        right_eye_xmin=coords[2]-15
        right_eye_ymin=coords[3]-15
        right_eye_xmax=coords[2]+15
        right_eye_ymax=coords[3]+15
        left_eye_coords=image[left_eye_ymin:left_eye_ymax,left_eye_xmin:left_eye_xmax]
        right_eye_coords=image[right_eye_ymin:right_eye_ymax,right_eye_xmin:right_eye_xmax]
        eye_coords=[[left_eye_xmin,left_eye_ymin,left_eye_xmax,left_eye_ymax],[right_eye_xmin,right_eye_ymin,right_eye_xmax,right_eye_ymax]]
        
        #cv2.rectangle(image,(left_eye_xmin,left_eye_ymin),(left_eye_xmax,left_eye_yamx),(0,0,255))
        #cv2.rectangle(image,(right_eye_xmin,right_eye_ymin),(right_eye_xmax,right_eye_yamx),(0,0,255))
        #cv2.imwrite("frame_l.jpg",left_eye_coords)
        #cv2.imwrite("frame_r.jpg",right_eye_coords)
        
        #raise NotImplementedError
        return left_eye_coords,right_eye_coords,eye_coords
    

    #def check_model(self):
        #raise NotImplementedError

    def preprocess_input(self, image):
        image_processed=cv2.resize(image,(self.input_shape[3],self.input_shape[2]))
        image_processed=image_processed.transpose((2,0,1))
        image_processed=image_processed.reshape(1,*image_processed.shape)
        #print(image_processed.shape)
        return image_processed
        #raise NotImplementedError

    def preprocess_output(self, outputs,width,height):
       
        coord=outputs[0]
        lef_eye_x=(coord[0].tolist()[0][0])*width
        lef_eye_y=(coord[1].tolist()[0][0])*height
        rig_eye_x=(coord[2].tolist()[0][0])*width
        rig_eye_y=(coord[3].tolist()[0][0])*height
        
        return lef_eye_x,lef_eye_y,rig_eye_x,rig_eye_y
       
    
        #raise NotImplementedError
'''def main(args):
     model='/home/workspace/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml'
     device=args.device
     image='/home/workspace/landmarks-regression-retail-0009.png'
     
     pd= FacialLandmarkDetection(model, device)
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
