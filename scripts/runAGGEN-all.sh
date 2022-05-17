#!/bin/bash
if [ $# -lt 3 ]
  then
      echo "[ERROR]: You must specify a {seed}
      and if you want the plain dataset(0) or perhaps shuffle the data(1)
      or even balance the classes(2) and the CrossType(0=BLX,1=ARITHMETIC).
      Lastly the selection for cross  method, 0 for randoms, 1 for randoms
      and keep the best parent and 2 for only cross the best parent. (Reminder,
      we always keep elitism, the best from before is maintained)"
      echo "[Ex.]: ./script.sh 150421 0 1 2"
    exit
fi
if [ -z "$1" ]
  then
    echo "[ERROR]: Couldn't read the seed value for some reason"
    echo "[Ex.]: ./script.sh 150421 0 1 2"
    exit
fi

if [ -z "$2" ]
  then
    echo "[ERROR]: Couldn't read the shuffle value for some reason"
    echo "[Ex.]: ./script.sh 150421 0 1 2"
    exit
fi

if [ -z "$3" ]
  then
    echo "[ERROR]: Couldn't read the cross value for some reason"
    echo "[Ex.]: ./script.sh 150421 0 1 2"
    exit
fi

if [ -z "$4" ]
  then
    echo "[ERROR]: Couldn't read the selection value for some reason"
    echo "[Ex.]: ./script.sh 150421 0 1 2"
    exit
fi

echo "[START-1]: Genetic GEN algorithm evaluation on ionosphere.arrf"
./../bin/AGGEN ionosphere.arff b g 1 $1 $2 $3 30 $4 0
echo "[START-2]: Genetic GEN algorithm evaluation on parkinson.arrf"
./../bin/AGGEN parkinsons.arff 1 2 1 $1 $2 $3 30 $4 0
echo "[START-3]: Genetic GEN algorithm evaluation on spectf-heart.arrf"
./../bin/AGGEN spectf-heart.arff 1 2 1 $1 $2 $3 30 $4 0

