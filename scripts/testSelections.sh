if [ $# -lt 1 ]
  then
      echo "[ERROR]: You must specify a {seed}"
      echo "[Ex.]: ./script.sh 150421"
    exit
fi

for i in 0 1
do
    for j in 0 1 2
    do
        ./runAGGEN-all.sh $1 0 $i $j
    done
done
