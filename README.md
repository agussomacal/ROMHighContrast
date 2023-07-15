# Reduced order modeling for elliptic problems with high contrast diffusion coefficients

1. This repository has all the code implementation of the paper 
[Reduced order modeling for elliptic problems with high contrast diffusion coefficients](https://hal.archives-ouvertes.fr/hal-03549810/document)

2. The [notebook](https://github.com/agussomacal/ROMHighContrast/blob/main/src/notebooks/InverseProblemPipeline.ipynb) for the practical session of CEMRACS 2023 [Linear and Nonlinear Schemes for Forward Model Reduction and Inverse Problems](http://smai.emath.fr/cemracs/cemracs23/summer-school.html)

To run the code on binder or Colab without setting up locally any environment go to:
- [![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/agussomacal/ROMHighContrast/main?labpath=%2Fsrc%2Fnotebooks%2FHighContrast.ipynb).
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/agussomacal/ROMHighContrast/blob/main/src/notebooks/InverseProblemPipeline.ipynb)

## Abstract
We consider a parametric elliptic PDE with a scalar piecewise constant diffusion coefficient taking arbitrary positive values  on fixed subdomains. This problem is not uniformly elliptic, as the contrast can be arbitrarily high, contrarily to the Uniform Ellipticity Assumption (UEA) that is commonly made on parametric elliptic PDEs. We construct reduced model spaces that approximate uniformly well all solutions with estimates in relative error that are independent of the contrast level. These estimates are sub-exponential in the reduced model dimension, yet exhibiting the curse of dimensionality as the number of subdomains grows. Similar estimates are obtained for the Galerkin projection, as well as for the state estimation and parameter estimation inverse problems.  A key ingredient in our construction and analysis is the study of the convergence towards limit solutions of stiff problems when diffusion tends to  infinity in certain domains.


# Setup for developers
We recommend first to work in a virtual environment which can be created using 
previously installed python packages venv or virtualenv through
```
python3.8 -m venv venv
```
or
```
virtualenv -p python3.8 test
```

Then activate virtual environment
```
. .venv/bin/activate
```
Install required packages usually through:
```
pip install -r requirements.txt 
```
However, if this doesn't work for you, try to install them one by one in the order specified by the requirements.txt file.


### Jupyter notebooks
In order to be able to run jupyter notebooks:
```
pip install ipykernel
python -m ipykernel install --user --name venv --display-name "venv"
jupyter nbextension enable --py --sys-prefix widgetsnbextension
```
Source: https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments 

   
