# Dive into Caustics: 3D Reconstruction of Refractive Caustics

<img src="https://github.com/YueYin27/nerfstudio_translucent/blob/master/media/oracle_method.png?raw=true" width="100%"/>

## Table of Contents

[Code Description](#1-code-description)

[Data Preparation](#2-data-preparation)

[Instructions for Running the Code](#3-instructions-for-running-the-code)
   - [Setup the environment](#1-setup-the-environment)
   - [Train the model](#2-train-the-model)
   - [Evaluate the model](#3-evaluate-the-model)
   - [Render the result as a video](#4-render-the-result-as-a-video)

[Acknowlegdements](#4-acknowledgements)

## 1. Data Preparation

You can download our [dataset](https://github.com/YueYin27/nerfstudio_for_engn8501/tree/main/caustics_bowl_pattern) online from our GitHub repo.

Or, you can install Blender, design a 3D model and run the following command to generate your own dataset:
```
Blender modelname.blend --python dataset_customization/view_train.py -b
Blender modelname.blend --python dataset_customization/view_val.py -b
Blender modelname.blend --python dataset_customization/view_test.py -b
```
Replace the `modelname.blend` with the name of your model.

## 2. Instructions for Running the Code

#### 1) Setup the environment

To run the code, you must first set up the environment for [nerfstudio](https://github.com/nerfstudio-project/nerfstudio).
Detailed instructions can be found [here](https://github.com/nerfstudio-project/nerfstudio#1-installation-setup-the-environment). The installation will take about 40 minutes.

*Note:* The environment configuration on Windows can be a little tricky. One thing we need to be careful about is that since MSVC 2019 use X86 but nvcc use X64, we need to use the command:
```
"C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
```
to make your msvc change to x64 before any installation steps. The reason can be found [here](https://stackoverflow.com/questions/12843846/problems-when-running-nvcc-from-command-line%5B/url%5D).

#### 2) Train the model

After setting up the environment, the following command will train our model on our synthetic dataset. Keep ``--vis wandb \`` if you use the weights & biases platform to visualise the training process and delete it otherwise.
```
ns-train nerfacto --machine.device-type cuda \
                  --machine.num-devices 1 \
                  --experiment-name caustics \
                  --project-name nerfstudio-caustics \
                  --pipeline.model.background-color random \
                  --pipeline.model.proposal-initial-sampler uniform \
                  --pipeline.model.near-plane 0.05 \
                  --pipeline.model.far-plane 15 \
                  --pipeline.model.num_nerf_samples_per_ray 256 \
                  --pipeline.datamanager.camera-optimizer.mode off \
                  --pipeline.model.use-average-appearance-embedding False \
                  --pipeline.model.distortion-loss-mult 0.1 \
                  --pipeline.model.disable-scene-contraction True \
                  --vis wandb \
         blender-depth-data \
                  --scale-factor 0.1 \
                  --data data/customized/caustics_bowl_pattern/
```

You can use ```ns-train -help ``` to learn more about the command and adjust the hyperparameter settings.

#### 3) Evaluate the model

After the training is finished, you can use the following command to evaluate the model you get(Replace the `OUTPUT_ROOT_PATH` variable with the file path you get from training).
```
OUTPUT_ROOT_PATH=outputs/caustics/nerfacto/2023-11-02_194437

ns-eval --load-config $OUTPUT_ROOT_PATH/config.yml \
        --output-path $OUTPUT_ROOT_PATH/output_test.json
```
### 4) Render the result as a video
Given a pre-trained model checkpoint, you can start the viewer by running
```
ns-viewer --load-config {outputs/.../config.yml}
```
First, we must create a path for the camera to follow. This can be done in the viewer under the "RENDER" tab. Orient your 3D view to the location where you wish the video to start, then press "ADD CAMERA". This will set the first camera keyframe. Continue to new viewpoints, adding additional cameras to create the camera path. 

Once finished, press "RENDER", which will display a modal that contains the command needed to render the video. Create a new terminal and run the command to generate the video.

## 4. Acknowledgements

Our project is built on the [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) framework.
