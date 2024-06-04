#!/bin/bash

# Loop over 10 integers
for i in {11..13}
do
    cp data/configs/lr_run_1.ini data/configs/lr_run_${i}.ini
done

sed -i -e 's/0.1/0.05/g' data/configs/lr_run_2.ini
sed -i -e 's/0.1/0.04/g' data/configs/lr_run_3.ini
sed -i -e 's/0.1/0.03/g' data/configs/lr_run_4.ini
sed -i -e 's/0.1/0.02/g' data/configs/lr_run_5.ini
sed -i -e 's/0.1/0.01/g' data/configs/lr_run_6.ini
sed -i -e 's/0.1/0.009/g' data/configs/lr_run_7.ini
sed -i -e 's/0.1/0.008/g' data/configs/lr_run_8.ini
sed -i -e 's/0.1/0.007/g' data/configs/lr_run_9.ini
sed -i -e 's/0.1/0.006/g' data/configs/lr_run_10.ini
sed -i -e 's/0.1/0.5/g' data/configs/lr_run_11.ini
sed -i -e 's/0.1/1/g' data/configs/lr_run_12.ini
sed -i -e 's/0.1/2/g' data/configs/lr_run_13.ini