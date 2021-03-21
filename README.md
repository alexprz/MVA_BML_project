# MVA – Bayesian Machine Learning – Project F1
## Authors
### Louis Maestrati, Quentin Spinat, Alexandre Pérez

## Download and install
To download the project and install its dependencies, please run the following command line commands in the directory of your choice.

```sh
git clone git@github.com:alexprz/MVA_BML_project.git
cd MVA_BML_project
python3 -m venv venv
source venv/bin/activate
pip install pip --upgrade
pip install -r requirements.txt
```

Note that our code has been checked for Python 3.9.2.

## Structure
The structure of our code is organized as follows.

Auxiliary files:
* `model.py` Implement a fully connected linear neural network with variable depth, width, input dimensions and output dimensions. The model is used in all experiments.
* `eoc.py` Implement functions related to finding points on the Edge Of Chaos (EOC), inspired by the code of [1].

Experiment files:
* `XP1.py` Reproduce the figure 5b of article [1] with custom activation functions. Note that we used tensorboard to monitor and plot the metrics.
* `XP2.py` Reproduce the figure 1 of article [1] with custom activation functions.
* `XP3.py` Study the evolution of the loss as a function of points on the EOC.

## Reproduce our experiments
<!-- ### As in our article -->
Note that some experiments use Tensorboard to monitor the results. The command `tensorboard --logdir tb_logs` allows to access the results by going to the given address in your browser (*e.g.* http://localhost:6006/).

**Figure 1a:**
```python
python3 XP1.py --nlayers 200 --nplayers 300 --act relu --sigb 0 --sigw 1.414
python3 XP1.py --nlayers 200 --nplayers 300 --act relu --sigb 1 --sigw 1
tensorboard --logdir tb_logs
```
**Figure 1b:**
```python
python3 XP1.py --nlayers 200 --nplayers 300 --act lrelu --ns 0.5 --sigb 0 --sigw 1.227
python3 XP1.py --nlayers 200 --nplayers 300 --act lrelu --ns 0.5 --sigb 0 --sigw 1
tensorboard --logdir tb_logs
```

**Figure 2a:**
```python
python3 XP2.py --nlayers 200 --nplayers 300 --act relu --sigb 1 --sigw 1
```

**Figure 2b:**
```python
python3 XP2.py --nlayers 200 --nplayers 300 --act relu --sigb 0 --sigw 1.414
```

**Figure 2c:**
```python
python3 XP2.py --nlayers 200 --nplayers 300 --act relu --sigb 0 --sigw 2
```

**Figure 3:**
```python
python3 XP2.py --nlayers 200 --nplayers 300 --act relu --sigb 0 --sigw 1.414
python3 XP2.py --nlayers 200 --nplayers 300 --act lrelu --ns 0.5 --sigb 0 --sigw 1.265
python3 XP2.py --nlayers 200 --nplayers 300 --act elu --sigb 0.2 --sigw 1.227
tensorboard --logdir tb_logs
```

**Figure 4a:**
```python
python3 XP3.py --nlayers 200 --nplayers 300 --act elu --sig_max 0.1 --epochs 10
```

**Figure 4b:**
```python
python3 XP3.py --nlayers 200 --nplayers 300 --act elu --sig_max 1 --epochs 10
```


<!-- ### With small computations to check code correctness -->


## References
[1] Hayou, S., Doucet, A., & Rousseau, J. (n.d.). On the Impact of the Activation Function on Deep Neural Networks Training.
