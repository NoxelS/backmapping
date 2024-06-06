#!/bin/bash

# Loop over 10 integers
for i in {12..25}
do
    cp data/configs/lr_run_1.ini data/configs/lr_run_${i}.ini
done

sed -i -e 's/LR/0.005/g' data/configs/lr_run_12.ini
sed -i -e 's/LR/0.004/g' data/configs/lr_run_13.ini
sed -i -e 's/LR/0.003/g' data/configs/lr_run_14.ini
sed -i -e 's/LR/0.002/g' data/configs/lr_run_15.ini
sed -i -e 's/LR/0.001/g' data/configs/lr_run_16.ini
sed -i -e 's/LR/0.0009/g' data/configs/lr_run_17.ini
sed -i -e 's/LR/0.0008/g' data/configs/lr_run_18.ini
sed -i -e 's/LR/0.0007/g' data/configs/lr_run_19.ini
sed -i -e 's/LR/0.0006/g' data/configs/lr_run_20.ini
sed -i -e 's/LR/0.0005/g' data/configs/lr_run_21.ini
sed -i -e 's/LR/0.00025/g' data/configs/lr_run_22.ini
sed -i -e 's/LR/0.0001/g' data/configs/lr_run_23.ini
sed -i -e 's/LR/0.00005/g' data/configs/lr_run_24.ini
sed -i -e 's/LR/0.00001/g' data/configs/lr_run_25.ini