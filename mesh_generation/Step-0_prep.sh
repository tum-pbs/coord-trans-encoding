
tar -xvf gen_airfoils.tar.gz

gfortran conv_for_cfl3d_plot3d.f90 -o conv_for_cfl3d_plot3d
gfortran conv_metric_to_p3d.f90 -o conv_metric_to_p3d
