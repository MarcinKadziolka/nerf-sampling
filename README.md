# Accelerating sampling in NeRF

One of the most prominent issues of the original [NeRF](https://www.matthewtancik.com/nerf) is a tedious two-stage sampling process that makes it unusable for real-time rendering.

This project aims to swap the vanilla hierarchical sampling process with a novel depth network. 

DepthNet allows to skip the first stage of sampling (which requires 64 samples and network queries) by predicting the most important area in one pass.

Example scenes rendered using 64 samples less per ray than original NeRF:


<img src="https://github.com/user-attachments/assets/afe29473-2e6e-4887-bde8-1a491dd7f974" alt="Hotdog" width="400"/>
<img src="https://github.com/user-attachments/assets/35e3fdc5-dbe9-412e-b46a-335691dee69b" alt="Ship" width="400"/>


The idea of the DepthNet is to approximate the most important area of the scene, by minimizing the distance between the point with the highest weight and the predicted depth. After training, an arbitrary method of sampling can be used to populate the region.

<img src="https://github.com/user-attachments/assets/b6d3a950-01e8-4c7e-b95c-4fb0de954c58" alt="Depth net optimization" width="600"/>

Visualizing predicted points in the 3D space gives accurate point clouds of the scenes:

<img src="https://github.com/user-attachments/assets/200db4e3-9b1f-4faa-8927-9680a48e78f8" alt="Lego point cloud" width="600"/>
<img src="https://github.com/user-attachments/assets/c685b607-7c14-488a-af27-bef77bb6cbd3" alt="Drums point cloud" width="600"/>

The network architecture consists of linear layers with skip-connections that process three inputs: the origin of the ray, the direction of the ray, and finally the same information encoded as the points of intersection between the ray and the conceptualized sphere around the object:

<img src="https://github.com/user-attachments/assets/25ea8de8-51e7-43e4-bc52-308310c735ce" alt="Network architecture" width="600"/>

# Dataset
NeRF synthetic dataset can be found here:

 [Nerf Synthetic Dataset | Kaggle](https://www.kaggle.com/datasets/nguyenhung1903/nerf-synthetic-dataset?resource=download) (accessed: 15.09.2024)

**Provide dataset folders in the following way:**

nerf-sampling/

        nerf_sampling/

                dataset/

                        example_dataset/

                        chair/

                        lego/

**Store pretrained NeRF models and DepthNets as follows:**

nerf-sampling/

        nerf_sampling/

                pretrained/

                        nerf/

                                dataset_name/

                                        200000.tar

                        depth_net/

                                dataset_name/

                                        files/

                                                sampler_experiment/

                                                        200000.tar

# Training

## Virtual environment

In order to run the scripts the virtual environment has to be created. 

Go to the `nerf-sampling` directory and run:

`conda env create -f environment.yml`

Activate the environment:

`conda activate nerf_sampling`

## Running the scripts

Now training and rendering can begin.

All following commands must be run from `nerf-sampling/nerf_sampling` directory

**To train new DepthNet run the following command:**

`python3 experiments/run.py -d dataset_name`

where dataset_name is the name of the folder put in the correct file structure.

**To render and test DepthNet run:**

`python3 experiments/render.py -d dataset_name`

**To plot the point clouds:**

`python3 experiments/plot.py`

The paths to the saved data of the scene have to be provided manually inside the `plot.py` file.

**Additional information**

 Read all the possibilities of training, rendering and options by running:

`python3 experiments/render.py --help`

`python3 experiments/run.py --help`

                               
