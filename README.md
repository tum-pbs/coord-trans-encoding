<style TYPE="text/css">
code.has-jax {font: inherit; font-size: 100%; background: inherit; border: inherit;}
</style>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'] // removed 'code' entry
    }
});
MathJax.Hub.Queue(function() {
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});
</script>
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-AMS_HTML-full"></script>

# coord-trans-encoding
This is the source code for the paper ["Towards high-accuracy deep learning inference of compressible turbulent flows over aerofoils"](https://arxiv.org/abs/2109.02183) by Liwei Chen and Nils Thuerey.

Additional information: [project page](https://ge.in.tum.de/2021/09/07/high-accuracy-transonic-rans-flow-predictions-with-deep-neural-networks-preprint-online-now/)

## Abstract:
The present study investigates the accurate inference of Reynolds-averaged Navier-Stokes solutions for the compressible flow over aerofoils in two dimensions with a deep neural network. Our approach yields networks that learn to generate precise flow fields for varying body-fitted, structured grids by providing them with an encoding of the corresponding mapping to a canonical space for the solutions. We apply the deep neural network model to a benchmark case of incompressible flow at randomly given angles of attack and Reynolds numbers and achieve an improvement of more than an order of magnitude compared to previous work. Further, for transonic flow cases, the deep neural network model accurately predicts complex flow behaviour at high Reynolds numbers, such as shock wave/boundary layer interaction, and quantitative distributions like pressure coefficient, skin friction coefficient as well as wake total pressure profiles downstream of aerofoils. The proposed deep learning method significantly speeds up the predictions of flow fields and shows promise for enabling fast aerodynamic designs.

# Data Generation

**Requirements**

- [CFL3D Version 6.7](https://nasa.github.io/CFL3D/) for data generation. The official source code is [here](https://github.com/NASA/CFL3D), but we have [our own repo with some changes](https://github.com/Hypersonichen/CFL3D).
- [Construct2D](https://sourceforge.net/projects/construct2d/) Note: we found "gfortran 9.4.0" causes problems so eventually used "gfortran 7.5.0" to build the executable code. Note also that one should use python2.7 to run the visualization script provided by Construct2D, e.g. `python2.7 postpycess.py`.
- [PyTorch](https://pytorch.org/) *tested with "1.6.0", with "1.10.2", with "1.11.0+cu113"*. We recommend installing via conda, e.g., with `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`. 

**Build CFL3D**

Clone the NASA's CFL3D code to your local computer: 

`git clone https://github.com/Hypersonichen/CFL3D.git`

`cd CFL3D/build/`

`./setup.sh`

In the current project, we only use the serial version "cfl3d_seq" in the folder "CFL3D/build/cfl/seq/".

**Mesh Generation**

`cd mesh_generation`

Step 0:
`./Step-0_prep.sh`

Step 1:
Modify Line #11 in "Step-1_script_meshing_v2.sh" with the correct path (where construct2d is installed), and run

`./Step-1_script_meshing_v2.sh` 

(Note: the script calls "construct2d". It might take minutes generating 1500 mesh files. According to the data files for airfoil coordinates, if the training edge is sharp, the script generates C-type grids; if blunt, it writes O-type grids. The output files are *.p3d, *_stats.p3d and *.nmf files.)

Step 2:

`./Step-2_check_otype.sh` (Note: this step checks o-type mesh files, and moved those into the folder "o-mesh_raw".)

Step 3:

`./Step-3_conv_for_cfl3d_plot3dgrid.sh` (Note: because "construct2d" uses ASCII plot3d format, we need to translate the ASCII files into unformated binary format for NASA's cfl3d code. We only need *.bin files for the simulation.)


**Run CFL3D**

`cd BASIC_simulations/`

`./prep.sh`

Modify Line # 42 in "dataGen.py" and make sure the file path is correct (i.e. the location of "cfl3d_seq"). 

`python dataGen.py`

The script reads the cfl3d input template `input_template.inp` and replaces the user-defined keywords with specified values. In the current version, we run 16000 iterations, and then average the results over another 8000 iterations. The final results, $\rho$, $\rho u$, $\rho v$, $\rho E$, are saved in npz format in the folder `train_avg`.

Note that potentially we can use well-tuned multi-grid method to accelerate the simulation. More technical details can be found in [CFL3D official website](https://nasa.github.io/CFL3D/). 

**Obtaining the metrics**

The ["metrics"](https://nasa.github.io/CFL3D/Cfl3dv6/V5Manual/GenCoor.pdf) contain geometrical information. As each airfoil only has one mesh and there is no mesh deformation, we can ouput the coordinate transformation metrics as a separate post-processing task. (Warning: this is a special case, in some unsteady projects where mesh deformation or aeroelasticity are involved, the metrics should be saved during the simulation. We will update this in the up-coming tutorial and project.) 

We want to re-build cfl3d (note: suppose we have copied the code "cfl3d_seq" built previously to somewhere, e.g. /usr/local/bin/)

`cd CFL3D/build/`

`make scruball` (note: this means all the previously-built files are deleted.)

`./metric_setup.sh` (we only use the serial version "cfl3d_seq" in the folder "CFL3D/build/cfl/seq/".)

`cd metric_generation`

`python metricGen.py` 

The script calls cfl3d_seq. Instead of running a flow simulation, it only reads mesh files and calculates "metrics", and then saves 7 variables (because it is 2D), i.e. `si4, sj1, sj3, sj4, sk1, sk3, sk4` to files.



# Training

We recommend using the pre-generated data for training.
- [Training set](https://drive.google.com/file/d/16z1ZL60yWyVfyvFU8Uq3SMVfMHEQkGJ_/view?usp=sharing) 2.24Gb. There are 970 airfoils and each airfoil case has two randomly-selected flow conditions, so totally we have 1940 flowfields.
- [Test set](https://drive.google.com/file/d/1fCYnhxfXicwxtlivItU6VJ_ltcuD51FI/view?usp=sharing) 23.7Mb. There are 20 airfoils that haven't been seen in the training set; each has one flow condition, and totally we have 20 flowfields.

Download and extract the data. Then, make sure the sub-folders "train_avg" and "test_avg" are under the folder "BASIC_data_coordinates_final_metricsAll_1940".

`nohup python runTrain.py > output.log` (Note: in Line 33 of "runTrain.py", if we use "expo=7", the number of trainable parameters is 30928388. Training runs 1290 epochs.)

One can also download the trained model as well as *pickle files [here](https://drive.google.com/file/d/1E7QC5BtwF0EI0r6OaUvB5qzDTLBLJwS2/view?usp=sharing), the random seed used in the model is "2125985365".

# Testing

Extract the "test_mesh.tar.gz".

`python cp_subplots.py`

`python skinfriction_subplots.py`

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

