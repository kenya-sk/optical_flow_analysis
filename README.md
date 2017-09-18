# flow

## workflow
### install openCV
```
pyenv install anaconda3-4.2.0  #should not be python3.6
pyenv local anaconda3-4.2.0
conda install -c https://conda.anaconda.org/menpo opencv3
```

### exec
1. draw_flow.py
    * caluculate optical flow and mean/variance
2. batch.py
    * process many files at ones
3. plot_graph.py
    * plot graphs of mean and variance by optical flow
4. get_frameList.py
    * get the frame number when event occurred 
5. luminance.py
    * get luminance by differential between two frames


## info
### help command
   * if you need argument information. please press [program name] -h .
