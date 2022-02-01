# CS x476 - Fall 2021

# Project 3: Scene Recognition with Deep Learning
## Refer jupyter notebook for info on dataset
## Setup:

1. Install Miniconda. (If you already have Miniconda installed, you can skip this step)
2. Create a conda environment using the appropriate terminal and command.
    ● On **Windows** , open the installed "Anaconda Powershell Prompt".
    ● On **MacOS** and **Linux** , you can open a terminal window.
    ● Modify and run the command in the terminal, replace the “<OS>” in the following
       command with your OS (Linux, Mac, Windows): conda env create -f
       cv_proj3_configs/cv_proj3_env_<OS>.yml
3. Check if the cv_proj3 environment has been created properly.
    ● Run: conda env list
4. Activate the conda environment.
    ● Run: conda activate cv_proj
    ● To deactivate it, run: conda deactivate
5. Install the project packages.
    ● Run: pip install -e. inside the repo folder.
    ● This is a good practice when setting up a new conda environment that may have pip
       requirements. It installs the repo as a package in the environment.
6. Open the jupyter notebook to work on the project.
    ● Run: jupyter notebook ./cv_proj3_code/proj3.ipynb
