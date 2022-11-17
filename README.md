# SAFE-Python
Python version of the Sensitivity Analysis for Everybody (SAFE) Toolbox.

### BEFORE STARTING

An introduction to the SAFE Toolbox is provided in the paper:

Pianosi, F., Sarrazin, F., Wagener, T. (2015), A Matlab toolbox for Global Sensitivity Analysis, Environmental Modelling & Software, 70, 80-85. The paper is freely available at: www.sciencedirect.com/science/article/pii/S1364815215001188

We recommend reading this (short) paper before getting started. Other reading materials, including general introductions to Sensitivity Analysis and case study applications, can be found at: www.safetoolbox.info

### INSTALLING THE PACKAGE

Download the SAFE-Python repository to your computer. You then have two options for installation. 

Option 1: from python command line (e.g. Anaconda prompt), go into the SAFE-Python folder and execute:

    pip install .

Option 2: from command line go into the SAFE-Python folder and execute:

	python -m pip install .


Notes for Windows users: python cannot be called directly from Windows command line. You have to go into the folder in which python is installed and then execute:

	python -m pip install mydir\SAFE-Python

(mydir is the directory in which the SAFE-Python folder is saved, and it shoud not contain which spaces)


NOTE: if you want to install the package without administrator rights, you may have to use:
     
	pip install --user mydir\SAFE-Python


### GETTING STARTED

To get started using SAFE, we suggest opening one of the workflow scripts in the 'workflow' folder and running the code step by step. The header of each workflow script gives a short description of the method and case study model, and of the main steps and purposes of that workflow, as well as references for further reading. The name of each workflow is composed as: workflow_method_model

Implemented models are:
- the hydrological Hymod model 
- the hydrological HBV model 
- the Ishigami and Homma test function 
- the Sobol' g-function 

Implemented methods are:
- eet (elementary effects test, or method of Morris)
- fast (Fourier amplitude sensitivity test)
- pawn
- rsa (regional sensitivity analysis)
- vbsa (variance-based sensitivity analysis, or method of Sobol')

Furthermore, SAFE includes additional workflow scripts:
- external: how to connect SAFE to a model running outside python
- tvsa: how to apply GSA methods to perform time-varying sensitivity analysis 
- visual: how to use visualisation functions for qualitative GSA

If the user still has no clear idea of what method(s) to start with, we suggest one of the three most widely used methods: eet (e.g. workflow_eet_hymod), rsa (workflow_rsa_hymod), vbsa (workflow_vbsa_hymod) or the visualization workflow (workflow_visual_ishigami_homma.m).
