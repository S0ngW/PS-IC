#!/bin/bash

export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

batchs=20
k=25

for i in $(seq 1 $batchs)
do
    (
        for j in $(seq 1 $k)
        do
            julia -t 1 simulation.jl $i $j $k
        done
    ) &
done

wait