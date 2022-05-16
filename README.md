# Optical Flow Analysis
This repository calculates statistics value of movement by using saprse optical flow. 

## Workflow
```
# 1. Build docker environment
docker build -t optical_flow:latest .
docker run -it optical_flow /bin/bash

# 2. Compile file needed for execution.
make all

# 3. Run the compiled file and enter various paths 
# (input video file path, output statistics value path and output video path) 
# on the command line.
./main

# 4. Create a time series graph of calculated statistics.
python plot_data.py
```

## Demo
![output](https://user-images.githubusercontent.com/30319295/56096043-f7f17e80-5f1d-11e9-88c8-ab04a54dcf59.gif)

## Graph of Statistics Value
<img src="./data/graph/mean.jpg" alt="mean" vspace="25">
<img src="./data/graph/var.jpg" alt="var" vspace="25">
<img src="./data/graph/max.jpg" alt="max" vspace="25">
