#!/bin/bash

# Loop from 26 to 38
for i in {2..25}
do
    cp data/configs/do_run_1.ini data/configs/do_run_${i}.ini
    sed -i -e s/do_run_1/do_run_${i}/g data/configs/do_run_${i}.ini
done

sed -i -e 's/DROPOUT/0.02/g' data/configs/do_run_1.ini
sed -i -e 's/DROPOUT/0.04/g' data/configs/do_run_2.ini
sed -i -e 's/DROPOUT/0.06/g' data/configs/do_run_3.ini
sed -i -e 's/DROPOUT/0.08/g' data/configs/do_run_4.ini
sed -i -e 's/DROPOUT/0.10/g' data/configs/do_run_5.ini
sed -i -e 's/DROPOUT/0.12/g' data/configs/do_run_6.ini
sed -i -e 's/DROPOUT/0.14/g' data/configs/do_run_7.ini
sed -i -e 's/DROPOUT/0.16/g' data/configs/do_run_8.ini
sed -i -e 's/DROPOUT/0.18/g' data/configs/do_run_9.ini
sed -i -e 's/DROPOUT/0.20/g' data/configs/do_run_10.ini
sed -i -e 's/DROPOUT/0.22/g' data/configs/do_run_11.ini
sed -i -e 's/DROPOUT/0.24/g' data/configs/do_run_12.ini
sed -i -e 's/DROPOUT/0.26/g' data/configs/do_run_13.ini
sed -i -e 's/DROPOUT/0.28/g' data/configs/do_run_14.ini
sed -i -e 's/DROPOUT/0.30/g' data/configs/do_run_15.ini
sed -i -e 's/DROPOUT/0.32/g' data/configs/do_run_16.ini
sed -i -e 's/DROPOUT/0.34/g' data/configs/do_run_17.ini
sed -i -e 's/DROPOUT/0.36/g' data/configs/do_run_18.ini
sed -i -e 's/DROPOUT/0.38/g' data/configs/do_run_19.ini
sed -i -e 's/DROPOUT/0.40/g' data/configs/do_run_20.ini
sed -i -e 's/DROPOUT/0.42/g' data/configs/do_run_21.ini
sed -i -e 's/DROPOUT/0.44/g' data/configs/do_run_22.ini
sed -i -e 's/DROPOUT/0.46/g' data/configs/do_run_23.ini
sed -i -e 's/DROPOUT/0.48/g' data/configs/do_run_24.ini
sed -i -e 's/DROPOUT/0.50/g' data/configs/do_run_25.ini