#!/bin/bash
if [ $# -lt 5 ]
  then
      echo "[ERROR]: You must specify a {seed}
      and if you want the plain dataset(0) or perhaps shuffle the data(1)
      or even balance the classes(2) and the type of Cross (0=BLX,1=ARITHMETIC).
      Also we need to know how often we do the localsearch (positive int) and to
      what porcentage of the population (float [0,1]) and whether we do it
      random(0) or to the best only (1)"
      echo "[Ex.]: ./script.sh 150421 0 1 10 0.1 0"
    exit
fi
if [ -z "$1" ]
  then
    echo "[ERROR]: Couldn't read the seed value for some reason"
      echo "[Ex.]: ./script.sh 150421 0 1 10 0.1 0"
    exit
fi

if [ -z "$2" ]
  then
    echo "[ERROR]: Couldn't read the shuffle value for some reason"
      echo "[Ex.]: ./script.sh 150421 0 1 10 0.1 0"
    exit
fi

if [ -z "$3" ]
  then
    echo "[ERROR]: Couldn't read the cross value for some reason"
      echo "[Ex.]: ./script.sh 150421 0 1 10 0.1 0"
    exit
fi

if [ -z "$4" ]
  then
    echo "[ERROR]: Couldn't read how often we do the local search for some reason"
      echo "[Ex.]: ./script.sh 150421 0 1 10 0.1 0"
    exit
fi

if [ -z "$5" ]
  then
    echo "[ERROR]: Couldn't read the percentage of population we search for some reason"
      echo "[Ex.]: ./script.sh 150421 0 1 10 0.1 0"
    exit
fi

if [ -z "$6" ]
  then
    echo "[ERROR]: Couldn't read whether we search randoms or only the best for some reason"
      echo "[Ex.]: ./script.sh 150421 0 1 10 0.1 0"
    exit
fi

echo "[START-1]: Genetic EST algorithm evaluation on ionosphere.arrf"
./../bin/AGEST ionosphere.arff b g 1 $1 $2 $3 10 1 $4 $5 $6
echo "[START-2]: Genetic EST algorithm evaluation on parkinson.arrf"
./../bin/AGEST parkinsons.arff 1 2 1 $1 $2 $3 10 1 $4 $5 $6
echo "[START-3]: Genetic EST algorithm evaluation on spectf-heart.arrf"
./../bin/AGEST spectf-heart.arff 1 2 1 $1 $2 $3 10 1 $4 $5 $6

