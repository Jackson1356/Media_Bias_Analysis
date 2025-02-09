#!/bin/bash

# Define the outlet
outlet="nypost2021"

# Define start and end index pairs
indexes=(
    "0 20000"
    "20000 45000"
    "45000"
    # Add more pairs as needed
)

# Loop through index pairs and execute the Python script with each one
for index_pair in "${indexes[@]}"
do
    # Split the pair into separate variables
    set -- $index_pair
    start_index=$1
    end_index=$2

    if [ -z "$end_index" ]; then
	    echo "Running Python script for $outlet starting from $start_index without an end index"
	    python main.py -n "$outlet" -s "$start_index"
    else
	    echo "Running Python script for $outlet from $start_index to $end_index"
	    python main.py -n "$outlet" -s "$start_index" -e "$end_index"
    fi
done

echo "All ranges have been processed."
