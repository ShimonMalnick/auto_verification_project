# auto_verification_project
This repo contains the code for the project in the course "Automatic Verification of Systems" (0368-4178 TAU) taken in the Fall of 2021-2022.\
The repository is based on the repositories of [Marabou](https://github.com/NeuralNetworkVerification/Marabou) and [FGP](https://github.com/klasleino/fast-geometric-projections) and is licensed accordingly.\
This repo can be used to verify local robustness of PyTorch Fully connected networks with ReLU activations by using the Marabou framework.
# Get started
First clone the repository of  [Marabou](https://github.com/NeuralNetworkVerification/Marabou):

```
cd auto_verification_project
git clone https://github.com/NeuralNetworkVerification/Marabou
```

and build it under its  [Build and Dependencies](https://github.com/NeuralNetworkVerification/Marabou#build-and-dependencies) instructions.
Then make sure you have python >= 3.7 installed with pip to install the additional dependencies (note that these requirements are relevant for reproducing results and some requirements may be relevant for just using the pytorch tool):
```
cd auto_verification_project
pip install -r requirements.txt
```
To use this repository just as a tool for running verification of PyTorch models, go to the [Verify PyTorch Using Marabou](#verify-pytorch-using-marabou) section. 
# Reproduce Results for

# Verify PyTorch Using Marabou
