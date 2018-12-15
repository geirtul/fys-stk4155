# FYS-STK4155 - Applied Data Analysis and Machine Learning
## Project 1
Description of the project is available here:
[Project 1 PDF](http://compphysics.github.io/MachineLearning/doc/Projects/2018/Project1/pdf/Project1.pdf)
<br />
## Project 2
Description of the project is available here:
[Project 2 PDF](http://compphysics.github.io/MachineLearning/doc/Projects/2018/Project2/pdf/Project2.pdf)
<br />

### Project structure
In the project2 folder you will find report/ and src/, where report/ contains the .tex and pdf of
the final report, and src/ contains the code. requirements.txt contains all needed packages to run
all the python code in this project.

#### src/
Within the src/ folder you will find the following:<br />
├── __src__ <br />
│   ├── __data__ <br />
│   ├── ising_data.py <br />
│   ├── lasso.py <br />
│   ├── logistic.py <br />
│   ├── __neural_net__ <br />
│   ├── ols.py <br />
│   ├── project2.py <br />
│   ├── resampling.py <br />
│   ├── ridge.py <br />
│   ├── task_ab.py <br />
│   └── task_c.py <br />

The data/ folder contains all the Ising model datafiles provided
by Mehta et al here: [Notebooks and data](https://physics.bu.edu/~pankajm/MLnotebooks.html)
<br />
The files in the src/ folder contain classes with regression methods with the same name. <br />
E.g ols.py -> orinary least squares, logistic.py -> logistic regression. <br/>
Exceptions to this are the following files: <br /><br />
├── ising_data.py (Generates 1D Ising data) <br />
├── task_ab.py (Produces results for task a and b in the project description) <br />
└── task_c.py (Produces results for task c in the project description) <br />
<br />
The neural_net/ folder contains all the files relevant to the neural network. <br />
├── __neural_net__ <br />
│   ├── network_testing.py (Deprecated)<br />
│   ├── neural_net.py (Object-oriented implementation of the network)<br />
│   └── run_network.py (Imports Ising data and runs the network)<br />
