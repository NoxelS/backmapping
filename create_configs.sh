#!/bin/bash

# Loop from 26 to 38
for i in {26..38}
do
    cp data/configs/lr_run_25.ini data/configs/lr_run_${i}.ini
    sed -i -e s/lr_run_25/lr_run_${i}/g data/configs/lr_run_${i}.ini
done

sed -i -e 's/0.00001/0.000009/g' data/configs/lr_run_26.ini
sed -i -e 's/0.00001/0.000008/g' data/configs/lr_run_27.ini
sed -i -e 's/0.00001/0.000007/g' data/configs/lr_run_28.ini
sed -i -e 's/0.00001/0.000006/g' data/configs/lr_run_29.ini
sed -i -e 's/0.00001/0.000005/g' data/configs/lr_run_30.ini
sed -i -e 's/0.00001/0.000004/g' data/configs/lr_run_31.ini
sed -i -e 's/0.00001/0.000003/g' data/configs/lr_run_32.ini
sed -i -e 's/0.00001/0.000002/g' data/configs/lr_run_33.ini
sed -i -e 's/0.00001/0.000001/g' data/configs/lr_run_34.ini
sed -i -e 's/0.00001/0.0000005/g' data/configs/lr_run_35.ini
sed -i -e 's/0.00001/0.0000001/g' data/configs/lr_run_36.ini
sed -i -e 's/0.00001/0.00000005/g' data/configs/lr_run_37.ini
sed -i -e 's/0.00001/0.00000001/g' data/configs/lr_run_38.ini