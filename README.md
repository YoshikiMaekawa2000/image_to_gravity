# image_to_gravity
## Overview
This repository presents a deep neural network which estimates a gravity direction from a single shot.
![overview](https://user-images.githubusercontent.com/37431972/107804047-c7dbc680-6da6-11eb-8035-eb043a23dd01.png)
## Datasets
Some datasets are available at [ozakiryota/dataset_image_to_gravity](https://github.com/ozakiryota/dataset_image_to_gravity).
## Usage
The following commands are just an example.  
Some trained models are available in image_to_gravity/keep.
### Regression
#### Training
```bash
$ cd ***/image_to_gravity/docker/docker
$ ./run.sh
$ cd regression
$ python3 train.py
```
#### Inference
```bash
$ cd ***/image_to_gravity/docker/docker
$ ./run.sh
$ cd regression
$ python3 infer.py
```
#### Inference with MC-Dropout
Preparing...
### MLE
#### Training
```bash
$ cd ***/image_to_gravity/docker/docker
$ ./run.sh
$ cd mle
$ python3 train.py
```
#### Inference
```bash
$ cd ***/image_to_gravity/docker/docker
$ ./run.sh
$ cd mle
$ python3 infer.py
```
#### Inference with MC-Dropout
Preparing...
## Citation
If this repository helps your research, please cite the paper below.  
```
Preparing...
```
The implementation used when it was published is available at Commit [2f66928](https://github.com/ozakiryota/image_to_gravity/tree/6ea94711b5ea6b7340856cf30a142b19d64b04d7).
## Related repositories
Preparing...
