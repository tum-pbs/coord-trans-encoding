#!/bin/bash
#OUT=./output.txt
 
## in memory here docs 
## thanks https://twitter.com/freebsdfrau
FILES=$(ls ./gen_airfoils/* )

                                                                                     
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
79
NWKE
25
QUIT
VOPT
RECD
1000000
QUIT
GRID
SMTH
QUIT
EOF
done 
