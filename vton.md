
# Virtual Cloth Try-On

This script sets up the environment and models needed to run the "Virtual Cloth Try-On" application. The process includes installing necessary dependencies, downloading models, and setting up Gradio for the user interface.

## 1. **Installing Required Packages**
```bash
!pip install --upgrade --no-cache-dir gdown
!pip install rembg[gpu] gradio==3.45.0
```
- **gdown**: Used for downloading files from Google Drive.
- **rembg**: A background removal tool that uses GPU acceleration.
- **gradio**: A Python library for building user interfaces for machine learning models. Here, it’s used to create the user interface for the Virtual Try-On application.

## 2. **Installing System Dependencies**
```bash
! sudo apt-get --assume-yes update
! sudo apt-get --assume-yes install build-essential libopencv-dev ...
```
This section installs various system libraries and dependencies required for running the models and the software stack:
- **build-essential**: Tools for compiling software.
- **libopencv-dev**: OpenCV development files for computer vision tasks.
- **libatlas-base-dev**: A library for optimized mathematical computations.
- **libprotobuf-dev, libleveldb-dev**: Libraries required for handling various model-related tasks.

## 3. **Downloading and Installing CMake**
```bash
! wget -c "https://github.com/Kitware/CMake/releases/download/v3.19.6/cmake-3.19.6.tar.gz"
! tar xf cmake-3.19.6.tar.gz
! cd cmake-3.19.6 && ./configure && make && sudo make install
```
CMake is a build system tool that is required for compiling the OpenPose repository. This command downloads CMake, extracts the files, and installs it on the system.

## 4. **Cloning OpenPose Repository**
```bash
! git clone --depth 1 -b "$ver_openpose" https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
```
- **OpenPose**: A library that detects human body, hand, face, and foot keypoints. This repository is cloned from GitHub to enable pose estimation.

## 5. **Setting Up OpenPose and Installing Dependencies**
```bash
! cd openpose && mkdir build && cd build
! cd openpose/build && cmake -DUSE_CUDNN=OFF -DBUILD_PYTHON=ON ..
! cd openpose/build && make -j`nproc`
```
This section sets up the OpenPose build environment, configures it using `cmake`, and then builds the software using `make`. OpenPose will be used for human pose estimation in the Virtual Try-On app.

## 6. **Cloning Virtual Try-On Repository**
```bash
!git clone https://github.com/practice404/clothes-virtual-try-on.git
os.makedirs("/content/clothes-virtual-try-on/checkpoints")
```
This command clones the repository containing the code for Virtual Cloth Try-On, where the core logic for the try-on process is stored.

## 7. **Downloading Pre-trained Models**
```bash
!gdown --id 18q4lS7cNt1_X8ewCgya1fq0dSk93jTL6 --output /content/clothes-virtual-try-on/checkpoints/alias_final.pth
```
Here, pre-trained models for different components of the Virtual Try-On system (e.g., segmentation, alias models) are downloaded and saved to the `checkpoints` directory.

## 8. **Installing Additional Libraries**
```bash
!pip install opencv-python torchgeometry
!pip install torchvision
```
Additional Python libraries are installed:
- **opencv-python**: For computer vision tasks.
- **torchgeometry**: A library for geometric transformations in PyTorch.
- **torchvision**: A package for image and video processing in PyTorch.

## 9. **Setting Up Self-Correction Human Parsing**
```bash
!git clone https://github.com/PeikeLi/Self-Correction-Human-Parsing
!mkdir checkpoints
!gdown --id 1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH
```
This repository helps with human parsing, which involves detecting the human body parts for virtual try-on. The pre-trained model weights are downloaded and stored in the `checkpoints` folder.

## 10. **Making Directories for Model Inputs**
```python
def make_dir():
  os.system("cd /content/ && mkdir inputs && cd inputs && mkdir test && cd test && mkdir cloth cloth-mask image image-parse openpose-img openpose-json")
```
This function creates necessary directories where inputs (cloth and human images) and outputs (processed images) will be stored.

## 11. **Running the Virtual Try-On Process**
```python
def run(cloth, model):
  make_dir()
  cloth.save("/content/inputs/test/cloth/cloth.jpg")
  model.save("/content/inputs/test/image/model.jpg")
  
  os.system("rm -rf /content/output/")
  os.system("python /content/clothes-virtual-try-on/run.py")
```
This function takes the uploaded cloth and human model images, saves them to disk, and runs the try-on process by executing the script in the cloned repository. The output is saved in the `/content/output/` folder.

## 12. **Building the User Interface with Gradio**
```python
import gradio as gr

with gr.Blocks() as app:
    with gr.Row():
        gr.Markdown("<center><h1> Clothes Virtual Try ON </h1><center>")
    with gr.Row():
        with gr.Column():
            cloth_input = gr.Image(sources=['upload'], type="pil", label="Upload the Cloth Image")
        with gr.Column():
            model_input = gr.Image(sources=['upload'], type="pil", label="Upload the Human Image")
    with gr.Row():
        final_output = gr.Image(sources=['upload'], type="pil", label="Final Prediction")
    with gr.Row():
        submit_button = gr.Button("Submit")

    submit_button.click(fn=run, inputs=[cloth_input, model_input],
                        outputs=[final_output])
app.launch(debug=True, share=True)
```
- **Gradio Interface**: This block of code creates a simple web interface for the user to upload images of the cloth and human model.
- **Buttons and Outputs**: Users can upload images, click the submit button, and get the final virtual try-on result.
- The app is launched with `app.launch(debug=True, share=True)`, which opens the interface in a new window.

---

### How to Run This
1. Install all the necessary libraries and dependencies.
2. Clone the repositories for OpenPose and the Virtual Try-On system.
3. Download the required pre-trained models.
4. Set up directories to store images and outputs.
5. Use Gradio to create a user-friendly interface to upload images and show the final virtual try-on result.

