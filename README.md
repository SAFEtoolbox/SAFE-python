# SAFEpython
Python version of the Sensitivity Analysis for Everybody (SAFE) Toolbox.

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL3.0-yellow.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)

<p align="center">
<img src="https://raw.githubusercontent.com/SAFEtoolbox/SAFEtoolbox.github.io/main/drawing2.png" width = "300px">
</p>

### BEFORE STARTING

An introduction to the SAFE Toolbox is provided in the paper:

Pianosi, F., Sarrazin, F., Wagener, T. (2015), A Matlab toolbox for Global Sensitivity Analysis, Environmental Modelling & Software, 70, 80-85. The paper is freely available at: https://doi.org/10.1016/j.envsoft.2015.04.009

We recommend reading this (short) paper before getting started. Other reading materials, including general introductions to Sensitivity Analysis and case study applications, can be found at: https://safetoolbox.github.io

### INSTALLING THE PACKAGE

#### Option 1: Installing the package using pip

``pip install safepython``

#### Option 2: Installing a local version of the package (if you want to customize the code)

Download the SAFEpython source code to your computer. You go into the SAFE-Python folder and execute: 

``pip install .``

#### Notes

- You can execute the previous commands from python command line (e.g. Anaconda prompt). 

- From command line, you should use:

option 1: ``python -m pip install safepython``
	
option 2: ``python -m pip install .``

- For windows users: python cannot be called directly from Windows command line. You have to go into the folder in which python is installed and then execute:

option 1: ``python -m pip install safepython``
	
option 2: ``python -m pip install mydir\SAFE-python``

(mydir is the directory in which the SAFEpython folder is saved, and it shoud not contain which spaces)


- If you want to install the package without administrator rights, you may have to use:
	
``pip install --user .``


### GETTING STARTED

To get started using SAFE, we suggest opening one of the workflow scripts in the 'examples' folder available in the [**_github repository_**](https://github.com/SAFEtoolbox/SAFE-python) and running the code step by step. The header of each workflow script gives a short description of the method and case study model, and of the main steps and purposes of that workflow, as well as references for further reading. The name of each workflow is composed as: workflow_method_model

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

### Note 

Please make sure that you download the version of the 'examples' folder that corresponds to the version of SAFEpython package you are using. To use the latest version of SAFEpython, you can update the package using: 

``pip install --upgrade safepython``

### HOW TO CITE SAFEPYTHON

If you would like to use the software, please cite it using the following:

Pianosi, F., Sarrazin, F., Wagener, T. (2015), A Matlab toolbox for Global Sensitivity Analysis, Environmental Modelling & Software, 70, 80-85, doi: 10.1016/j.envsoft.2015.04.009.

[![DOI](https://img.shields.io/badge/doi.org/10.1016/j.envsoft.2015.04.009-purple.svg)](https://doi.org/10.1016/j.envsoft.2015.04.009)
