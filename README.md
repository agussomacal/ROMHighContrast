# Reduced order modeling for elliptic problems with high contrast diffusion coefficients

This repository has all the code implementation of the paper 
[Reduced order modeling for elliptic problems with high contrast diffusion coefficients](https://hal.archives-ouvertes.fr/hal-03549810/document)

To run the code on binder without setting up locally any environment go to [![Binder]](https://mybinder.org/v2/gh/agussomacal/ROMHighContrast/main?labpath=%2Fsrc%2Fnotebooks%2FHighContrast.ipynb).

## Abstract
We consider a parametric elliptic PDE with a scalar piecewise constant diffusion coefficient taking arbitrary positive values  on fixed subdomains. This problem is not uniformly elliptic, as the contrast can be arbitrarily high, contrarily to the Uniform Ellipticity Assumption (UEA) that is commonly made on parametric elliptic PDEs. We construct reduced model spaces that approximate uniformly well all solutions with estimates in relative error that are independent of the contrast level. These estimates are sub-exponential in the reduced model dimension, yet exhibiting the curse of dimensionality as the number of subdomains grows. Similar estimates are obtained for the Galerkin projection, as well as for the state estimation and parameter estimation inverse problems.  A key ingredient in our construction and analysis is the study of the convergence towards limit solutions of stiff problems when diffusion tends to  infinity in certain domains.

