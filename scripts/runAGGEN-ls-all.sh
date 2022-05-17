#!/bin/bash
if [ $# -lt 6 ]
  then
      echo "[ERROR]: You must specify a {seed}
      and if you want the plain dataset(0) or perhaps shuffle the data(1)
      or even balance the classes(2) and the CrossType(0=BLX,1=ARITHMETIC).
      Lastly the selection for cross  method, 0 for randoms, 1 for randoms
      and keep the best parent and 2 for only cross the best parent. (Reminder,
      we always keep elitism, the best from before is maintained)
      Also we need to know how often we do the localsearch (positive int) and to
      what porcentage of the population (float [0,1]) and whether we do it
      random(0) or to the best only (1)"
      echo "[Ex.]: ./script.sh 150421 0 1 2 10 0.1 0"
    exit
fi
if [ -z "$1" ]
  then
    echo "[ERROR]: Couldn't read the seed value for some reason"
      echo "[Ex.]: ./script.sh 150421 0 1 2 10 0.1 0"
    exit
fi

if [ -z "$2" ]
  then
    echo "[ERROR]: Couldn't read the shuffle value for some reason"
      echo "[Ex.]: ./script.sh 150421 0 1 2 10 0.1 0"
    exit
fi

if [ -z "$3" ]
  then
    echo "[ERROR]: Couldn't read the cross value for some reason"
      echo "[Ex.]: ./script.sh 150421 0 1 2 10 0.1 0"
    exit
fi

if [ -z "$4" ]
  then
    echo "[ERROR]: Couldn't read the selection value for some reason"
      echo "[Ex.]: ./script.sh 150421 0 1 2 10 0.1 0"
    exit
fi

if [ -z "$5" ]
  then
    echo "[ERROR]: Couldn't read how often we do the local search for some reason"
      echo "[Ex.]: ./script.sh 150421 0 1 2 10 0.1 0"
    exit
fi

if [ -z "$6" ]
  then
    echo "[ERROR]: Couldn't read the percentage of population we search for some reason"
    echo "[Ex.]: ./script.sh 150421 0 1 2 10 0.1 0"
    exit
fi

if [ -z "$7" ]
  then
    echo "[ERROR]: Couldn't read whether we search randoms or only the best for some reason"
    echo "[Ex.]: ./script.sh 150421 0 1 2 10 0.1 0"
    exit
fi

echo "[START-1]: Doing local search in ionosphere.arrf"
./../bin/AGGEN ionosphere.arff b g 1 $1 $2 $3 10 $4 1 $5 $6 $7
echo "[START-2]: Doing local search in parkinson.arrf"
./../bin/AGGEN parkinsons.arff 1 2 1 $1 $2 $3 10 $4 1 $5 $6 $7
echo "[START-3]: Doing local search in spectf-heart.arrf"
./../bin/AGGEN spectf-heart.arff 1 2 1 $1 $2 $3 10 $4 1 $5 $6 $7

