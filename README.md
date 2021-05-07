# XAI Benchmark

A benchmark for comparing and evaluating local model-agnostic post-hoc explainers.

## Setup

Run the following command to install necessary dependencies. A symbolic link will be built to *xaibenchmark* allowing you to change the source code without reinstallation.

    pip install -r requirements.txt

To fetch the *adult* dataset, navigate into `data/adult/` and run

    python setup_adult.py

## Usage

Import the module using

    import xaibenchmark

