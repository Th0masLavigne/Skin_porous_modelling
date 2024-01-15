# *In vivo* data-driven poromechanical modelling of the apparent viscoelasticity of human skin under tension

This repository contains the codes used to generate the results presented in *Lavigne et al.*[^1]. This paper proposes a proof of concept for a poroelastic model to account to simulate the mechanical behaviour of human skin. A two-layer model has been considered and a calibration procedure has been performed to reproduce the apparent viscoelasticity of *in vivo* human skin, using a porous media approach.


## Archive organization
- [Experimental Data](./Experimental_Data/) contains the [excel file](./Experimental_Data/Healthy_skin.xlsx) with the imposed displacement and measured force acquired with the in-house device presented in *Jacquet et al.*[^3].
- [Mesh GMSH](./Mesh_GMSH/) contains the code used to generate the labelled mesh before curvature. 
- [FEniCSx](./FEniCSx/) contains the codes used for calibration and evaluation of the model.

## Versions
give versions of GMSH & FEniCSx
mettre lien vers sites
commande docker ?
parallel cpmputation
NLOPT


The version used of FEniCSx is v0.5.2. [Dockerfile](Dockerfile) and built images are made available. To pull the image, after having installed docker, run `th0maslavigne/dolfinx:v0.5.2`. Otherwise the image can be built by running `docker build .` in the folder of the DockerFile. Then the container can be interactively executed through the command below:

```sh
docker run -ti -v $(pwd):/home/fenicsx/shared -w /home/fenicsx/shared th0maslavigne/dolfinx:v0.5.2
```


## Experiment
The aim is to reproduce a traction-relaxation experiment carried out with the in-house device presented in *Jacquet et al.*[^3]. The experiment consists in four loading-sustaining-unloading cycles of 8mm amplitude. 

<img 
    style="display: block; 
           margin-left: auto;
           margin-right: auto;
           width: 50%;"
src=https://github.com/Th0masLavigne/Skin_porous_modelling/blob/main/images/exp.png
alt="Experiment">
</img>

## Geometry

A curved two-layer (cutis and subcutis) geometry has been considered. The dimensions were increased to limit the effects of the boundary conditions.

<img 
    style="display: block; 
           margin-left: auto;
           margin-right: auto;
           width: 50%;"
src=https://github.com/Th0masLavigne/Skin_porous_modelling/blob/main/images/geom.png
alt="Geometry">
</img>

## Model
The porous model is presented in *Lavigne et al.*[^1] and *Lavigne et al.*[^2].

 $$   \left(\frac{\varepsilon^l_i}{K^l_i}+\frac{1-\varepsilon^l_i}{K^s_i} \right)\frac{\mathrm{D}^s p^l}{\mathrm{D}\text{t}}+\nabla\cdot\mathbf{v} -\nabla\cdot\left(\frac{k^\varepsilon_i}{\mu^l}\mathbf{\nabla}p^l\right) = 0~\text{on }\Omega_i,~i\in[1,2]$$
 $$ \mathbf{\nabla}\cdot\mathbf{t}^{\text{tot}} = 0~\text{on }\Omega_i,~i\in[1,2]$$




[^1]: T. Lavigne, S. Urcun, E. Jacquet, J. Chambert, A. Elouneg, C. Suarez, S.P.A. Bordas, G. Sciumè, PY. Rohan, *In vivo* data-driven poromechanical modelling of the apparent viscoelasticity of human skin under tension
[^2]: T. Lavigne, S. Urcun, P-Y. Rohan, G. Sciumè, D. Baroli, S.P.A. Bordas, Single and bi-compartment poro-elastic model of perfused biological soft tissues: FEniCSx implementation and tutorial: implementation in FEniCSx, https://doi.org/10.1016/j.jmbbm.2023.105902
[^3]: E. Jacquet, S. Joly, J. Chambert, K. Rekik, P. Sandoz, Ultra-light extensometer for the assessment of the mechanical properties of the human skin in vivo ,https://doi.org/10.1111/srt.12367