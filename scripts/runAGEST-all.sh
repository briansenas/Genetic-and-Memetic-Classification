#!/bin/bash
if [ $# -lt 2 ]
  then
      echo "[ERROR]: You must specify a {seed}
      and if you want the plain dataset(0) or perhaps shuffle the data(1)
      or even balance the classes(2) and the CrossType(0=BLX,1=ARITHMETIC)."
      echo "[Ex.]: ./script.sh 150421 0 1"
    exit
fi
if [ -z "$1" ]
  then
    echo "[ERROR]: Couldn't read the seed value for some reason"
    echo "[Ex.]: ./script.sh 150421 0 1"
    exit
fi

if [ -z "$2" ]
  then
    echo "[ERROR]: Couldn't read the shuffle value for some reason"
    echo "[Ex.]: ./script.sh 150421 0 1"
    exit
fi

if [ -z "$3" ]
  then
    echo "[ERROR]: Couldn't read the cross value for some reason"
    echo "[Ex.]: ./script.sh 150421 0 1"
    exit
fi

echo "[START-1]: Genetic EST algorithm evaluation on ionosphere.arrf"
./../bin/AGEST ionosphere.arff b g 1 $1 $2 $3 30 0
echo "[START-2]: Genetic EST algorithm evaluation on parkinson.arrf"
./../bin/AGEST parkinsons.arff 1 2 1 $1 $2 $3 30 0
echo "[START-3]: Genetic EST algorithm evaluation on spectf-heart.arrf"
./../bin/AGEST spectf-heart.arff 1 2 1 $1 $2 $3 30 0

