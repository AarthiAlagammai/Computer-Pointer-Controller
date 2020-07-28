# Computer-Pointer-Controller
## Introduction
Computer pointer controller app is used to control the mouse pointer of the computer through gaze estimation.This app takes video as input and then app estimates eye-direction and head-pose and based on that estimation it move the mouse pointers.This project uses multiple models in the same machine and coordinate the flow of data between those models to move the mouse pointer

## Architecture Diagram

## Setup and Installation

  ### Openvino installation:
  To run the application in this tutorial, the OpenVINO™ toolkit and its dependencies must already be installed and verified using the included demos.
  The installion guide for
  [Windows](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_windows.html)
  [Linux](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html)
  [Mac](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_macos.html)
  
  Step 1:
    Clone this repository https://github.com/AarthiAlagammai/Computer-Pointer-Controller/
    
 Step 2:
    Initialize the openVINO environment:
    ```
    source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
    ```
 Step 3:
    Download the following models by using openVINO model downloader:
    1. Face Detection Model
    
     python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name face-detection-adas-binary-0001 --output_dir models
  
     
   2.Facial Landmarks Detection Model
   
      python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name landmarks-regression-retail-0009 --output_dir models
      
   
   
   3.Head Pose Estimation Model
   
   ```
   python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name head-pose-estimation-adas-0001 --output_dir models
   ```
   
   4.Gaze Estimation Model
   
   ```
   python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name gaze-estimation-adas-0002 --output_dir models```
   ```

## Run application on your local machine
Open a new terminal and run the following commands:-

1. Change the directory to src directory of project repository
  ```
  cd <project-repo-path>/src
   ```
   
2. Run the main.py file(CPU)
  ```
  python main.py -f <Path of xml file of face detection model> \
  -fl <Path of xml file of facial landmarks detection model> \
  -hp <Path of xml file of head pose estimation model> \
  -g <Path of xml file of gaze estimation model> \
  -i <Path of input video file or enter cam for taking input video from webcam> 
  ```
  If you want to run app on GPU:-
   ```
  python main.py -f <Path of xml file of face detection model> \
  -fl <Path of xml file of facial landmarks detection model> \
  -hp <Path of xml file of head pose estimation model> \
  -g <Path of xml file of gaze estimation model> \
  -i <Path of input video file or enter cam for taking input video from webcam> 
  -d GPU
   ```
If you want to run app on FPGA:-
 ```
  python main.py -f <Path of xml file of face detection model> \
  -fl <Path of xml file of facial landmarks detection model> \
  -hp <Path of xml file of head pose estimation model> \
  -g <Path of xml file of gaze estimation model> \
  -i <Path of input video file or enter cam for taking input video from webcam> 
  -d HETERO:FPGA,CPU
   ```
### NOTE:
If you get Unsupported media format error, then it means that C:\Program Files (x86)\IntelSWTools\openvino_2020.2.117\opencv\bin does not contain opencv_videoio_ffmpeg430_64.dll and some other dll file that we need to run inference on the demo video. Then you will find ffmpeg-download file(C:\Program Files (x86)\IntelSWTools\openvino_2020.2.117\opencv) in this path. Run this file with powershell and you issue will be resolved
