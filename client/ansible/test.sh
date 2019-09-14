#!/bin/bash
SET=$(seq 1 10)
for i in $SET
do
    ./train.sh $i
done
