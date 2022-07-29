
#!/bin/bash

#FILES=$(find /home/liwei/Simulations/thuerey_research/data/database/airfoil_database/ -type f -newermt '2018-01-17' ! -newermt '2018-01-18')
#FILES=$(find /home/liwei/Simulations/thuerey_research/data/database/airfoil_database/ -type f )

mkdir c-mesh

FILES=$(ls *.nmf )
i=0
for file in $FILES; do
    var="$(grep -c FARFIELD "${file}")"
#var="$(grep '<td><a href="http://www.blabla.cc' file.txt)"
#i=$((i+1))
if [[ $var -lt 3 ]]
then
    echo "${file//.nmf}"
#    mv ${file//.nmf}*.* o-mesh
else
./conv_for_cfl3d_plot3d    << EOF
${file//.nmf}
EOF
fi
done 
#./transform_shear_minus0.5    << EOF

mv *.bin c-mesh
