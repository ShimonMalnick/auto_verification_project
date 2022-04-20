# auto_verification_project
This repo contains a tool for verifying PyTorch DNN's using the [Marabou](https://github.com/NeuralNetworkVerification/Marabou)
framework. This was done as art of the project in the course "Automatic Verification of Systems" (0368-4178 TAU) taken in the Fall of 2021-2022.\
The repository can also be used to reproduce some results from the projects (project can be seen [here](project_writeup.pdf)).
The repository is based on the repository of [Marabou](https://github.com/NeuralNetworkVerification/Marabou) and 
contains DNN's from the [FGP](https://github.com/klasleino/fast-geometric-projections) repository as these networks were used for comparison in the project.\
This repo can be used to verify local robustness of PyTorch Fully connected networks with ReLU activations by using the Marabou framework.
# Get started
First clone the repository of  [Marabou](https://github.com/NeuralNetworkVerification/Marabou):

```
cd auto_verification_project
git clone https://github.com/NeuralNetworkVerification/Marabou
```

and build it under its  [Build and Dependencies](https://github.com/NeuralNetworkVerification/Marabou#build-and-dependencies) instructions.
Then make sure you have python >= 3.7 installed with pip to install the additional dependencies:
```
cd auto_verification_project
pip install -r requirements.txt
```
To use this repository as a tool for running verification of PyTorch models, go to the [Verify PyTorch Using Marabou](#verify-pytorch-using-marabou) section.
# Verify PyTorch Using Marabou
To use this repo for verification of pytorch networks using Marabou, make sure your network consists only with 
[*Linear*](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) 
layers and [*ReLU*](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html) activations. The current script provides support for the network I created in the project. To load your
model add a corresponding function in [model_loaders.py](model_loaders.py) that returns the loaded model.
Then run:
```
python verify_pytorch_net.py --help
```
To see the documentation on the relevant parameters.\
To reproduce results for *My Own DNN* from the project run:
```
python verify_pytorch_net.py both 784
```