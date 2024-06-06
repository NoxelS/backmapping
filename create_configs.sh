#!/bin/bash

# Loop over 10 integers
for i in {2..14}
do
    cp data/configs/fe_run_1.ini data/configs/fe_run_${i}.ini
    sed -i -e s/fe_run_1/fe_run_${i}/g data/configs/fe_run_${i}.ini

done
sed -i -e 's/UNITS/2/g' data/configs/fe_run_1.ini
sed -i -e 's/UNITS/4/g' data/configs/fe_run_2.ini
sed -i -e 's/UNITS/6/g' data/configs/fe_run_3.ini
sed -i -e 's/UNITS/8/g' data/configs/fe_run_4.ini
sed -i -e 's/UNITS/16/g' data/configs/fe_run_5.ini
sed -i -e 's/UNITS/32/g' data/configs/fe_run_6.ini
sed -i -e 's/UNITS/64/g' data/configs/fe_run_7.ini
sed -i -e 's/UNITS/128/g' data/configs/fe_run_8.ini
sed -i -e 's/UNITS/256/g' data/configs/fe_run_9.ini
sed -i -e 's/UNITS/512/g' data/configs/fe_run_10.ini
sed -i -e 's/UNITS/1024/g' data/configs/fe_run_11.ini
sed -i -e 's/UNITS/2048/g' data/configs/fe_run_12.ini
sed -i -e 's/UNITS/4096/g' data/configs/fe_run_13.ini
sed -i -e 's/UNITS/8192/g' data/configs/fe_run_14.ini