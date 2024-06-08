#!/bin/bash

# Loop from 26 to 38
for i in {2..25}
do
    cp data/configs/do_run_1.ini data/configs/do_run_${i}.ini
    sed -i -e s/do_run_1/do_run_${i}/g data/configs/do_run_${i}.ini
    sed -i -e s/DROPOUT/0.${i*2}/g data/configs/do_run_26.ini
done

# sed -i -e 's/DROPOUT/0.000008/g' data/configs/do_run_27.ini
# sed -i -e 's/DROPOUT/0.000007/g' data/configs/do_run_28.ini
# sed -i -e 's/DROPOUT/0.000006/g' data/configs/do_run_29.ini
# sed -i -e 's/DROPOUT/0.000005/g' data/configs/do_run_30.ini
# sed -i -e 's/DROPOUT/0.000004/g' data/configs/do_run_31.ini
# sed -i -e 's/DROPOUT/0.000003/g' data/configs/do_run_32.ini
# sed -i -e 's/DROPOUT/0.000002/g' data/configs/do_run_33.ini
# sed -i -e 's/DROPOUT/0.000001/g' data/configs/do_run_34.ini
# sed -i -e 's/DROPOUT/0.0000005/g' data/configs/do_run_35.ini
# sed -i -e 's/DROPOUT/0.0000001/g' data/configs/do_run_36.ini
# sed -i -e 's/DROPOUT/0.00000005/g' data/configs/do_run_37.ini
# sed -i -e 's/DROPOUT/0.00000001/g' data/configs/do_run_38.ini