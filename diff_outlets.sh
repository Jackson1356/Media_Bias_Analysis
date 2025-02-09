#!/bin/bash

# Define an array of configurations
configs=("alternet2018" "alternet2019" "alternet2020" "alternet2021")

# Loop through the configurations and execute the Python script with each one
for config in "${configs[@]}"
do
	echo "Running Python script with configuration: $config"
	python main.py -n "$config"
done
echo "All configurations have been processed."

