#!/bin/bash
#OUT=./output.txt
 
## in memory here docs 
## thanks https://twitter.com/freebsdfrau
FILES=$(ls ./gen_airfoils/* )

# When using TEST:
# FILES=$(ls ./gen_airfoils_TEST/* )

                                                                                     
for file in $FILES; do
echo "${file}"
/home/liwei/Codes/Construct2D_2.1.4/construct2d <<EOF
${file}
VOPT
JMAX
129
QUIT
SOPT
NSRF
81
NWKE
24
LESP
1e-3
QUIT
VOPT
RECD
5000000
QUIT
GRID
SMTH
QUIT
EOF
done 
