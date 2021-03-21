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
* `model.py` Implement a fully connected linear neural network with variable depth, width, activation, input dimensions and output dimensions. The model is used in all experiments.
* `eoc.py` Implement functions related to finding points on the Edge Of Chaos (EOC), inspired by the code of [1].

Experiment files:
* `XP1.py` Reproduce the figure 5b of article [1] with custom activation functions. Note that we used TensorBoard to monitor and plot the metrics.
* `XP2.py` Reproduce the figure 1 of article [1] with custom activation functions.
* `XP3.py` Study the evolution of the loss as a function of points on the EOC.

## Reproduce our experiments
The following commands permit to reproduce the figures of our report.

Note that some experiments use TensorBoard to monitor the results. The command `tensorboard --logdir tb_logs` allows to access the results by going to the given address in your browser (*e.g.* http://localhost:6006/).

Note also that most of the experiments require heavy computations (except for figure 2). Lighter versions are proposed in the next section.

**Figure 1a:** (Heavy)
```python
python3 XP1.py --epochs 100 --nlayers 200 --nplayers 300 --act relu --sigb 0 --sigw 1.414
python3 XP1.py --epochs 100 --nlayers 200 --nplayers 300 --act relu --sigb 1 --sigw 1
tensorboard --logdir tb_logs
```
**Figure 1b:** (Heavy)
```python
python3 XP1.py --epochs 100  --nlayers 200 --nplayers 300 --act lrelu --ns 0.5 --sigb 0 --sigw 1.265
python3 XP1.py --epochs 100  --nlayers 200 --nplayers 300 --act lrelu --ns 0.5 --sigb 0 --sigw 1
tensorboard --logdir tb_logs
```

**Figure 2a, 2b and 2c:** (Light)
```python
python3 XP2.py --nlayers 200 --nplayers 300 --act relu --sigb 1 --sigw 1
python3 XP2.py --nlayers 200 --nplayers 300 --act relu --sigb 0 --sigw 1.414
python3 XP2.py --nlayers 200 --nplayers 300 --act relu --sigb 0 --sigw 2
```

**Figure 3:** (Heavy)
```python
python3 XP1.py --epochs 100 --nlayers 200 --nplayers 300 --act relu --sigb 0 --sigw 1.414
python3 XP1.py --epochs 100 --nlayers 200 --nplayers 300 --act lrelu --ns 0.5 --sigb 0 --sigw 1.265
python3 XP1.py --epochs 100 --nlayers 200 --nplayers 300 --act elu --sigb 0.2 --sigw 1.227
tensorboard --logdir tb_logs
```

**Figure 4a:** (Heavy)
```python
python3 XP3.py --epochs 10 --nlayers 200 --nplayers 300 --act elu --sigb_max 0.1
```

**Figure 4b:** (Heavy)
```python
python3 XP3.py --epochs 10 --nlayers 200 --nplayers 300 --act elu --sigb_max 1
```

## Run small experiments for sanity check
Most of our experiments take time to run (even on GPU support), to launch some "cheap" runs of our code with small computation costs, you can use the following commands. Note that the results produced by these sanity checks have no interest but to show correctness of our code.


**Experiment 1:** (used for figure 1 and 3)
```python
python3 XP1.py --epochs 1 --act relu --sigb 0 --sigw 1.414
python3 XP1.py --epochs 1 --act lrelu --ns 0.5 --sigb 0 --sigw 1.265
python3 XP1.py --epochs 1 --act elu --sigb 0.2 --sigw 1.227
tensorboard --logdir tb_logs
```

**Experiment 2:** (used for figure 2)

This experiment is fast to run and is thus the same as the one in our report.
```python
python3 XP2.py --nlayers 200 --nplayers 300 --act relu --sigb 1 --sigw 1
python3 XP2.py --nlayers 200 --nplayers 300 --act relu --sigb 0 --sigw 1.414
python3 XP2.py --nlayers 200 --nplayers 300 --act relu --sigb 0 --sigw 2
```

**Experiment 3:** (used for figure 4)
```python
python3 XP3.py --epochs 1 --act elu --nsigb 2 --sigb_max 0.1
```

## Reference
[1] Hayou, S., Doucet, A., & Rousseau, J. (n.d.). On the Impact of the Activation Function on Deep Neural Networks Training.
