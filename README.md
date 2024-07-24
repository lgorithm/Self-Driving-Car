# Self-Driving Car Project

This project implements a self-driving car using end-to-end learning. The model is trained to predict steering angles from images of the road. The implementation is inspired by the research papers "End to End Learning for Self-Driving Cars" by Nvidia and "End to End Learning based Self-Driving using JacintoNet."

## Table of Contents

- [Overview](#overview)
- [Research Papers](#research-papers)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Overview

This project demonstrates the use of deep learning techniques to develop a self-driving car. The model is trained to predict steering angles directly from images of the road, eliminating the need for manually designed features. The project leverages convolutional neural networks (CNNs) to learn the driving task in an end-to-end manner.

## Research Papers

The implementation is based on the following research papers:

- [End to End Learning for Self-Driving Cars by Nvidia](https://arxiv.org/pdf/1604.07316.pdf)
- [End to End Learning based Self-Driving using JacintoNet](https://ieeexplore.ieee.org/document/8576190)

## Dataset 

I used the dataset provided by Sully Chen for training and evaluation. The dataset contains images and corresponding steering angles.

- Dataset: [Autopilot-TensorFlow Dataset](https://github.com/SullyChen/driving-datasets)

## Installation

To get started with the project, clone the repository and install the required dependencies:

`git clone https://github.com/lgorithm/Self-Driving-Car.git`
`cd self-driving-car`

## Usage

To use the trained model for inference, run the following command:

`python run.py`

To visualize training using Tensorboard use tensorboard --logdir=./logs, then open http://0.0.0.0:6006/ into your web browser.

## Acknowledgements

This project is inspired by the following research papers:

- [End to End Learning for Self-Driving Cars by Nvidia](https://arxiv.org/pdf/1604.07316.pdf)
- [End to End Learning based Self-Driving using JacintoNet](https://ieeexplore.ieee.org/document/8576190)

I'm also inspired by the implementation of Sully Chen and the dataset provided by him:

- Dataset: [Autopilot-TensorFlow Dataset](https://github.com/SullyChen/Autopilot-TensorFlow)
