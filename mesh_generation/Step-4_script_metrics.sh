
#!/bin/bash

FILES=$(ls ./metric_vol_cfl3d/ )
#FILES=$(ls ../test_metric/ )
#FILES=$(ls ../test_variants_metric/ )



#FILES=$(ls ./train_metric/* ) # this is the wrong one :)
i=0
for file in $FILES; do
    echo "${file//_metric.bin}"
#    mv ${file//.nmf}*.* o-mesh
#./conv_metric_to_p3d_TEST    << EOF
#./conv_metric_to_p3d_TEST_variants    << EOF
./conv_metric_to_p3d    << EOF
${file//_metric.bin}
EOF
done 
