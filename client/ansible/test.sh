#!/bin/bash
start=${s}
end=${e}
server=${server}
epoch=${epoch}
model=${model}

while [ $# -gt 0 ]; do
   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        # echo $1 $2 // Optional to see the parameter:value result
   fi
  shift
done

if [ -z $start ]; then
  echo '$start is not set'
  exit 1
fi

if [ -z $end ]; then
  echo '$end is not set'
  exit 1
fi

if [ -z $server ]; then
  echo '$server is not set'
  exit 1
fi

if [ -z $epoch ]; then
  echo '$epoch is not set'
  exit 1
fi

SET=$(seq $start $end)
for i in $SET
do
    ./train.sh --round $i --server $server --epoch $epoch --model $model
done
