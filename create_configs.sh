#!/bin/bash

# Loop from 26 to 38
for i in {2..13}
do
    cp data/configs/relu_run_1.ini data/configs/relu_run_${i}.ini
    sed -i -e s/relu_run_1/relu_run_${i}/g data/configs/relu_run_${i}.ini
done

sed -i -e 's/ALPHA/0.001/g' data/configs/relu_run_1.ini
sed -i -e 's/ALPHA/0.005/g' data/configs/relu_run_2.ini
sed -i -e 's/ALPHA/0.010/g' data/configs/relu_run_3.ini
sed -i -e 's/ALPHA/0.015/g' data/configs/relu_run_4.ini
sed -i -e 's/ALPHA/0.020/g' data/configs/relu_run_5.ini
sed -i -e 's/ALPHA/0.030/g' data/configs/relu_run_6.ini
sed -i -e 's/ALPHA/0.040/g' data/configs/relu_run_7.ini
sed -i -e 's/ALPHA/0.050/g' data/configs/relu_run_8.ini
sed -i -e 's/ALPHA/0.060/g' data/configs/relu_run_9.ini
sed -i -e 's/ALPHA/0.070/g' data/configs/relu_run_10.ini
sed -i -e 's/ALPHA/0.080/g' data/configs/relu_run_11.ini
sed -i -e 's/ALPHA/0.090/g' data/configs/relu_run_12.ini
sed -i -e 's/ALPHA/0.100/g' data/configs/relu_run_13.ini


for i in {2..15}
do
    cp data/configs/aug_run_1.ini data/configs/aug_run_${i}.ini
    sed -i -e s/aug_run_1/aug_run_${i}/g data/configs/aug_run_${i}.ini
done

sed -i -e 's/ANGLE/0/g' data/configs/aug_run_1.ini
sed -i -e 's/ANGLE/2/g' data/configs/aug_run_2.ini
sed -i -e 's/ANGLE/4/g' data/configs/aug_run_3.ini
sed -i -e 's/ANGLE/6/g' data/configs/aug_run_4.ini
sed -i -e 's/ANGLE/8/g' data/configs/aug_run_5.ini
sed -i -e 's/ANGLE/10/g' data/configs/aug_run_6.ini
sed -i -e 's/ANGLE/15/g' data/configs/aug_run_7.ini
sed -i -e 's/ANGLE/20/g' data/configs/aug_run_8.ini
sed -i -e 's/ANGLE/25/g' data/configs/aug_run_9.ini
sed -i -e 's/ANGLE/50/g' data/configs/aug_run_10.ini
sed -i -e 's/ANGLE/60/g' data/configs/aug_run_11.ini
sed -i -e 's/ANGLE/80/g' data/configs/aug_run_12.ini
sed -i -e 's/ANGLE/100/g' data/configs/aug_run_13.ini
sed -i -e 's/ANGLE/140/g' data/configs/aug_run_14.ini
sed -i -e 's/ANGLE/180/g' data/configs/aug_run_15.ini