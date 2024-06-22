#!/bin/bash

# for i in {2..14}
circuit=$1

# Check if the user input is valid
if [[ $circuit != "qft" && $circuit != "random" ]]; then
    echo "Invalid circuit. Usage: ./bench.sh [qft/ramdom]"
    exit 1
fi

case $circuit in
    qft)
        for t in $(seq 2 2 14); do
            echo "Running for $t qubits"
            python3 run.py -c qft -n $t -o out/qft.csv
        done
    ;;
    random)
        for t in $(seq 2 2 14); do
            echo "Running for $t qubits"
            python3 run.py -c random -n $t -o out/random.csv
        done
    ;;
esac

# for t in $(seq 2 2 14); do
#     echo "Running for $t qubits"
#     python3 run.py -c random -n $t -o out/random.csv
# done
# # python3 run.py -c random -n 3 -o out/log.csv
# # python3 run.py -c random -n 4 -o out/log.csv