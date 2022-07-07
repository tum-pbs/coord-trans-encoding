# coord-trans-encoding
This is the source code for the paper ["Towards high-accuracy deep learning inference of compressible turbulent flows over aerofoils"](https://arxiv.org/abs/2109.02183) by Liwei Chen and Nils Thuerey.

Additional information: [project page](https://ge.in.tum.de/2021/09/07/high-accuracy-transonic-rans-flow-predictions-with-deep-neural-networks-preprint-online-now/)

## Abstract:
The present study investigates the accurate inference of Reynolds-averaged Navier-Stokes solutions for the compressible flow over aerofoils in two dimensions with a deep neural network. Our approach yields networks that learn to generate precise flow fields for varying body-fitted, structured grids by providing them with an encoding of the corresponding mapping to a canonical space for the solutions. We apply the deep neural network model to a benchmark case of incompressible flow at randomly given angles of attack and Reynolds numbers and achieve an improvement of more than an order of magnitude compared to previous work. Further, for transonic flow cases, the deep neural network model accurately predicts complex flow behaviour at high Reynolds numbers, such as shock wave/boundary layer interaction, and quantitative distributions like pressure coefficient, skin friction coefficient as well as wake total pressure profiles downstream of aerofoils. The proposed deep learning method significantly speeds up the predictions of flow fields and shows promise for enabling fast aerodynamic designs.

# Tutorial

**Requirements**

- [CFL3D Version 6.7](https://nasa.github.io/CFL3D/) for data generation. [Source code](https://github.com/NASA/CFL3D)
- [Construct2D](https://sourceforge.net/projects/construct2d/)
- [PyTorch](https://pytorch.org/) *tested with "1.6.0", with "1.10.2", with "1.11.0+cu113"*. We recommend installing via conda, e.g., with `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`. 

**Data Generation**

*** BASIC_data_coordinates_final_metricsAll ***

# Closing Remarks

If you find the approach useful, please cite our paper via:
```
@misc{https://doi.org/10.48550/arxiv.2109.02183,
  doi = {10.48550/ARXIV.2109.02183},
  url = {https://arxiv.org/abs/2109.02183},
  author = {Chen, Li-Wei and Thuerey, Nils},
  keywords = {Fluid Dynamics (physics.flu-dyn), Machine Learning (cs.LG), FOS: Physical sciences, FOS: Physical sciences, FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Towards high-accuracy deep learning inference of compressible turbulent flows over aerofoils},
  publisher = {arXiv},
  year = {2021},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

This work was supported by the ERC Consolidator Grant *SpaTe* (CoG-2019-863850).

