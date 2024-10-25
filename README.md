# Accelerating sampling in NeRF



## Dataset and model structure
NeRF synthetic dataset can be found ![here](https://www.kaggle.com/datasets/nguyenhung1903/nerf-synthetic-dataset?resource=download) (accessed: 15.09.2024)

**Provide datasets folders in the following way:**

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







                               
