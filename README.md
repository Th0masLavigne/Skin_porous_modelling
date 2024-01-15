# *In vivo* data-driven poromechanical modelling of the apparent viscoelasticity of human skin under tension

This repository contains the codes used to generate the results presented in *Lavigne et al.*[^1]. This paper proposes a proof of concept for a poroelastic model to account to simulate the mechanical behaviour of human skin. A two-layer model has been considered and a calibration procedure has been performed to reproduce the apparent viscoelasticity of *in vivo* human skin, using a porous media approach.


## Archive organization
- [Experimental Data](./Experimental_Data/)
- Mesh_gmsh code
- FEniCSx code

## Versions
give versions of GMSH & FEniCSx
mettre lien vers sites
commande docker ?
parallel cpmputation
NLOPT


The version used of FEniCSx is v0.5.2. Pandas were integrated for the visualization (excel and csv). [Dockerfile](Dockerfile) and built images are made available. To pull the image, after having installed docker, run `th0maslavigne/dolfinx:v0.5.2`. Otherwise the image can be built by running `docker build .` in the folder of the DockerFile. Then the container can be interactively executed through the command below:



## Experiment


## Geometry



## Model
ref a l'ancien article
*Lavigne et al.*[^2]

##




[^1]: T. Lavigne, S. Urcun, E. Jacquet, J. Chambert, A. Elouneg, C. Suarez, S.P.A. Bordas, G. Sciumè, PY. Rohan, *In vivo* data-driven poromechanical modelling of the apparent viscoelasticity of human skin under tension
[^2]: T. Lavigne, S. Urcun, P-Y. Rohan, G. Sciumè, D. Baroli, S.P.A. Bordas, Single and bi-compartment poro-elastic model of perfused biological soft tissues: FEniCSx implementation and tutorial: implementation in FEniCSx, https://doi.org/10.1016/j.jmbbm.2023.105902
