#!/bin/bash
### Line above tells to use the bash
### User fill in HERE
### If #SBATCH, not a comment but a directive for the slurm command.
###
### directives:
#
## Required batch arguments
#SBATCH --job-name=Opti2
#SBATCH --partition=CPU_Compute
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
##SBATCH --mem-per-cpu=32GB
#
## Suggested batch arguments
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thomas.lavigne@ensam.eu
#
## Logging arguments (IMPORTANT)
#SBATCH --output=slurm_%x-%j.out
#SBATCH --error=slurm_%x-%j.err

### Variables Summary
### echo: affiche à l'écran => écrit les informations du job dans le slurm_%x-%j.out
### tout ce qui commence par un "$" est une variable. 
echo ""
echo -e "\033[34m---------------------------------------------------------------------------------------\033[0m"
echo -e "\033[34mVariables Summary: \033[0m"
echo -e "\tWorking Directory = $SLURM_SUBMIT_DIR"
echo -e "\tJob ID = $SLURM_JOB_ID"
echo -e "\tJob Name = $SLURM_JOB_NAME"
echo -e "\tJob Hosts = $SLURM_JOB_NODELIST"
echo -e "\tNumber of Nodes = $SLURM_NNODES"
echo -e "\tNumber of Cores = $SLURM_NTASKS"
echo -e "\tCores per Node = $SLURM_NTASKS_PER_NODE"

### Modules
module purge
echo ""
echo -e "Purge"
module load EasyBuild
## module avail (lists the available modules)
module load foss
echo ""
echo -e "load foss"
## raccourci pour module load: ml, pour module avail ml av, module list 
module load FEniCSx/0.5.2
echo ""
echo -e "load FEniCSx"
# 
module load openpyxl/3.0.10
echo ""
echo -e "load openpyxl"
# 
module load gmsh
echo ""
echo -e "load gmsh"
# load matplotlib
module load matplotlib
echo ""
echo -e "load matplotlib"
# load NLopt
module load NLopt
echo ""
echo -e "load NLopt"
## print in the out the versions and available modules 
# module list
## help sur un module:
# module spider SciPy-bundle/2022.05
## pour voir les modules cachés
# module --show hidden avail
## Librairie NLOPT pour optimisation en python pratique et simple

# mkdir ./GMSH
# cd $SLURM_SUBMIT_DIR/GMSH
# python3 gmsh_file.py

### Execution en parallèle
cd $SLURM_SUBMIT_DIR
mkdir ./${SLURM_JOB_ID}
cp $SLURM_SUBMIT_DIR/functions.py $SLURM_SUBMIT_DIR/${SLURM_JOB_ID}/functions.py
cd $SLURM_SUBMIT_DIR/${SLURM_JOB_ID}
mpirun -n ${SLURM_NTASKS} python ../optimise2.py
#
### Write any other commands after if needed
# execute several one
# change to the other dorectory
# cd $SLURM_SUBMIT_DIR/directory2 
# mpirun -n ${SLURM_NTASKS} python file2.py
# ${SLURM_NTASKS} mettre en argument du fichier python si opti sequentiel/parallèle; par défaut mettre à 1
# https://koor.fr/Python/CodeSamples/SysArgv.wp

### Send Commands
## print the analysis
echo ""
echo -e "Analysis: compute seff $SLURM_JOB_ID"
## seff ${SLURM_JOB_ID}

### EOF