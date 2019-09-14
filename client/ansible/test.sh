#!/bin/bash
SET=$(seq 1 100)
for i in $SET
do
    ./train.sh $i 1
done
