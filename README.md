# Computer-Pointer-Controller
## Introduction
Computer pointer controller app is used to control the mouse pointer of the computer using the user's eye gaze.The app takes video as input to the app and then calculates the pose of the head which is then used to calculate the new mouse pointer.The app is build on edge using Intel Openvino tool kit.

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
  -l <CPU extension file if you are using Openvino version below 2020>

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
In Windows in some versions of CPU MFX version uninitialised or Unsupported media format error occurs which imples  C:\Program Files (x86)\IntelSWTools\openvino_2020.2.117\opencv\bin does not contain opencv_videoio_ffmpeg430_64.dll and some other dll file that we need to run inference on the demo video.
To overcome this error we have to run ffmeg-download file present opencv folder(file path (C:\Program Files (x86)\IntelSWTools\openvino_2020.2.117\opencv) .Run this file with powershell to overcome this issue.

## Documentation
Details about the used pretrained model:

1.[Face Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)

2.[Facial Landmarks Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)

3.[Head Pose Estimation Model]( https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)

4.[Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

Following are command line arguments that can use for while running the main.py file 

python main.py:-

-h :Help Get the information about all the command line arguments

-f (required) : Specify the path of Face Detection model's xml file

-hp (required) : Specify the path of Head Pose Estimation model's xml file

-fl (required) : Specify the path of Facial Lnadmark Estimation model's xml file

-g (required) : Specify the path of Gaze Estimation model's xml file

-i (required) : Specify the path of input video file or enter "CAM" for taking input video from webcam

-d (optional) : Specify the target device to infer the video file on the model. Suppoerted devices are: CPU, GPU, FPGA (For running on FPGA used HETERO:FPGA,CPU), MYRIAD.

-l (optional) : Specify the absolute path of cpu extension if some layers of models are not supported on the device.

-pt (optional) : Specify the probability threshold for face detection model to detect the face accurately from video frame.

-flags (optional) : Specify the flags from fd, fld, hp, ge if you want to visualize the output of corresponding models of each frame (write flags with space seperation. Ex:- -flags fd fld hp).


