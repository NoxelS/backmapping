#!/bin/bash

# Loop over 10 integers
for i in {2..3}
do
    cp data/configs/activation_run_1.ini data/configs/activation_run_${i}.ini
done

sed -i -e 's/activation_1/activation_2/g' data/configs/activation_run_2.ini
sed -i -e 's/linear/sigmoid/g' data/configs/activation_run_2.ini

sed -i -e 's/activation_1/activation_3/g' data/configs/activation_run_3.ini
sed -i -e 's/linear/tanh/g' data/configs/activation_run_3.ini