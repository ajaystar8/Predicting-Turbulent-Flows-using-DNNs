# Predictions of turbulent shear flows using deep neural networks

[![GitHub license](https://img.shields.io/badge/license-Apache-blue.svg)](
https://github.com/drkostas/COSC525-Project1/blob/master/LICENSE)

## README is under construction. To be updated soon ⚒️ 

## Table of Contents

+ [About](#about)
  + [Dataset](#dataset)
+ [TODO](#todo)
+ [License](#license)

## About <a name="about"></a> 

`TL;DR`

Created as part of my internship at Institute for Plasma Research, Gandhinagar, Ahmedabad.

We have implemented Deep Neural Network architectures as proposed by [Srinivasan et al.](https://arxiv.org/abs/1905.03634) and made an attempt to reproduce the results as described in the study.

`Detailed Explanation`

The project aims to establish a workflow for modeling the turbulent flow of fluids using DNNs. Specifically, Recurrent Neural Network (RNN) will be employed for the above stated task. 

The first task is to compare the performance of RNNs with Long Short Term Memory (LSTM) to quantify the superior capabilities of LSTMs in handling time series data, a crucial aspect of the project. 

Subsequently, the project will involve generating parameterized time-series data featuring [Moehlis coefficients](https://iopscience.iop.org/article/10.1088/1367-2630/6/1/056). Validation of this data is essential to ensure the absence of laminarization over time. 

The project’s final phase will encompass the training of the LSTM model using the generated dataset. The predictions made by this model will be validated against established physical metrics. This approach ensures that the model’s outputs align with Moehlis model which is a reduced dimensional solution of the Navier-Stokes equation governing the underlying fluid turbulence.



### Dataset <a name="dataset"></a> 

The datasets generated consists of the values of Moehlis coefficients at different points in time. 

These are derived by solving a set of nine ordinary differential equations. The amplitudes are obtained for 4001-time points, resulting in a time series data. The required number of datasets can be generated, depending on the input provided.

## TODO <a name="todo"></a>

Read the [TODO](TODO.md) to see the current task list. 

_The python script [Turbulent_Shear_Flow_Prediction.py](Turbulent_Shear_Flow_Prediction.py) is still under development. Until then please refer the [Notebook Version](Notebooks/Turbulent_Shear_Flow_Prediction_Notebook.ipynb)._

## License <a name = "license"></a>

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.
