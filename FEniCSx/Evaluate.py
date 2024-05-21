# Thomas Lavigne
# 21-03-2024
# 
# 
# 
from functions import *
# Additionnal libraries and functions needed
import time
import numpy as np
from dolfinx.io        import XDMFFile, gmshio	
from mpi4py            import MPI
from dolfinx.fem       import (FunctionSpace, Function, Constant, dirichletbc, locate_dofs_topological,
								Expression, form, assemble_scalar )
from petsc4py.PETSc    import ScalarType
from ufl               import (VectorElement, FiniteElement, MixedElement, TensorElement, Measure,
								split, dot, TestFunctions, nabla_div, grad, inner, Identity, TrialFunction,
								derivative, div, variable, sym)
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import math
# 
##############################################################
##############################################################
################## load the experimental data ################
##############################################################
##############################################################
# 
# Specify the file path and sheet name
file_path  = '../Experimental_Data/Healthy_skin.xlsx'
sheet_name = 'Test B - Sain'
# 
time_second, _ , _ , _ , _ , _ , displacement_milli_meter, force_newton = read_excel_sheet2(file_path, sheet_name)
# 
# Filter the noisy reaction force : 
n0 = time_second.index(30)
n1 = time_second.index(75)
n2 = time_second.index(90)
n3 = time_second.index(105)
n4 = time_second.index(145)
# 
# Depending on the step (loading-sustaining-unloading) apply the left-centered-right mean
force_newton[:n0]   = calculer_moyenne_g(50,force_newton[:n0])
force_newton[n0:n1] = calculer_moyenne_c(1000,force_newton[n0:n1])
force_newton[n1:n2] = calculer_moyenne_d(50,force_newton[n1:n2])
force_newton[n2:n3] = calculer_moyenne_g(50,force_newton[n2:n3])
force_newton[n3:n4] = calculer_moyenne_c(1000,force_newton[n3:n4])
# 
# Reduce the number of elements for simulation:
dt_objective  = 0.5
dt_experiment = time_second[1]-time_second[0]
nth           = int(dt_objective/dt_experiment)
# 
time_list = keep_every_n_elements(time_second,nth)
real_u_m  = [1e-3*elem/2 for elem in keep_every_n_elements(displacement_milli_meter,nth)]
RF_exp   = keep_every_n_elements(force_newton,nth)
# 
# 
# Correction of the pre-stress
# Identify the number of initial 'zeros'
Nzeros = int(10/dt_objective)-2
RF_exp[:Nzeros]=np.zeros(Nzeros) 
# 
# Identify the number of steps for the computation
num_steps = len(real_u_m)
#
#
##############################################################
##############################################################
########################## Computation #######################
##############################################################
##############################################################
# 
# Set time counter
begin_t = time.time()
# 
# initialize time
t=0
#------------------------------------------------------------#
#                   Load the Geometry from GMSH              #
#------------------------------------------------------------#
# 
filename = "../Mesh_GMSH/Mesh_SKIN.msh"
mesh, cell_tag, facet_tag = gmshio.read_from_msh(filename, MPI.COMM_WORLD, 0, gdim=3)
# The mesh has initially been created as a plane mesh. It is deformed to its curved 
# state here-after
radius   = 48e-3
zx,zy,zz = 1e-16, 1e-16, 1e-16
center_y = zy
# 
def du_y(x):
	theta = x[1]/(radius+x[2])*np.ones_like(x[0])
	return -x[1]+(radius+x[2])*np.sin(theta)
# 
def du_z(x):
	theta = x[1]/(radius+x[2])*np.ones_like(x[0])
	return -x[2]+(radius+x[2])*np.cos(theta)
# 
updated_mesh_space = FunctionSpace(mesh, mesh.ufl_domain().ufl_coordinate_element())
# Evaluation of the displacement to update the mesh 
d_u_               = Function(updated_mesh_space)
d_u_.interpolate(lambda x: np.stack((np.zeros_like(x[0]), du_y(x), du_z(x))))
d_u_.x.scatter_forward()
mesh.geometry.x[:,:mesh.geometry.dim] += d_u_.x.array.reshape((-1, mesh.geometry.dim))
# The resulting mesh can be observed in the following file
# with XDMFFile(MPI.COMM_WORLD, "Initial_mesh_tag.xdmf", "w") as xdmftag:
# 		xdmftag.write_mesh(mesh)
# 		xdmftag.write_meshtags(facet_tag, mesh.geometry)
# 		xdmftag.write_meshtags(cell_tag, mesh.geometry)
#
# Identify indices of the cells for each region for material definition
hypoderm_indices = [x for x in cell_tag.indices if (cell_tag.values[x] == 200)]
epiderm_indices  = [x for x in cell_tag.indices if (cell_tag.values[x] == 100)]
# Ensure all indices have been attributed
try :
	assert(len(cell_tag.indices) == len(epiderm_indices)+len(hypoderm_indices))
	if MPI.COMM_WORLD.rank       == 0:
		print("All cell tags have been attributed")
except:
	if MPI.COMM_WORLD.rank       == 0:
		print("*************") 
		print("Forgotten tags => material badly defined")
		print("*************") 
		exit()
# 
#------------------------------------------------------------#
#                   Function Spaces                          #
#------------------------------------------------------------#
# Parameter space
DG0_space = FunctionSpace(mesh, ("DG", 0))
# Mixed Space (R2,R) -> (u,p)
CG1_v     = VectorElement("CG", mesh.ufl_cell(), 1)
CG2       = VectorElement("CG", mesh.ufl_cell(), 2)
CG1       = FiniteElement("CG", mesh.ufl_cell(), 1)
# 
CG1v_space        = FunctionSpace(mesh, CG1_v)
CG2_space 	      = FunctionSpace(mesh, CG2)
CG1_space 		  = FunctionSpace(mesh, CG1)
MS                = FunctionSpace(mesh, MixedElement([CG2,CG1]))
# 
tensor_elem  = TensorElement("CG", mesh.ufl_cell(), degree=1, shape=(3,3))
tensor_space = FunctionSpace(mesh, tensor_elem)
# 
#------------------------------------------------------------#
#                   Material Definition                      #
#------------------------------------------------------------#
# 
# Solid scaffold
# Young Moduli [Pa]
E_cutis               = Constant(mesh, ScalarType(684250.4796883139)) 
E_subcutis            = Constant(mesh, ScalarType(47789.133894134284))  
# Possion's ratios [-]
nu_cutis              = Constant(mesh, ScalarType(0.48))
#  0.48 0.3
nu_subcutis           = Constant(mesh, ScalarType(0.3))
#
# Porous material
# Interstitial fluid viscosity [Pa.s]
viscosity             = Constant(mesh, ScalarType(5e-3))   
# Interstitial porosity [-]
porosity_cutis        = Constant(mesh, ScalarType(0.2))
porosity_subcutis     = Constant(mesh, ScalarType(0.4))
# Intrinsic permeabilitty [m^2]
permeability_cutis    = Constant(mesh, ScalarType(9.428584320870872e-15)) 
permeability_subcutis = Constant(mesh, ScalarType(5.029994538924334e-13)) 
# Fluid bulk modulus [Pa]
Kf                    = Constant(mesh, ScalarType(2.2e9))
# Solid bulk modulus [Pa]
Ks                    = Constant(mesh, ScalarType(1e10))
# Biot Coefficient [-]
beta                  = Constant(mesh, ScalarType(1))
# 
if MPI.COMM_WORLD.rank == 0:
		print(f"cutis : E={E_cutis.value}, k={permeability_cutis.value}; subcutis E={E_subcutis.value}, k={permeability_subcutis.value}")
# Map the porosity at the current time step
porosity                               = Function(DG0_space)
porosity.x.array[hypoderm_indices]     = np.full_like(hypoderm_indices, porosity_subcutis.value, dtype=ScalarType)
porosity.x.array[epiderm_indices]      = np.full_like(epiderm_indices, porosity_cutis.value, dtype=ScalarType)
porosity.x.scatter_forward()
# 
# Map the Porosity at the previous time step
porosity_n                             = Function(DG0_space)
porosity_n.x.array[:]                  = porosity.x.array[:]
porosity_n.x.scatter_forward()
# 
# Mapping the permeabilitty
permeability                               = Function(DG0_space)
permeability.x.array[hypoderm_indices]     = np.full_like(hypoderm_indices, permeability_subcutis.value, dtype=ScalarType)
permeability.x.array[epiderm_indices]      = np.full_like(epiderm_indices, permeability_cutis.value, dtype=ScalarType)
permeability.x.scatter_forward()
# 
# Storativity coefficient
S            = (porosity/Kf)+(1-porosity)/Ks
# 
# Lame coefficients
lambda_cutis			= Constant(mesh, ScalarType(E_cutis.value*nu_cutis.value/((1+nu_cutis.value)*(1-2*nu_cutis.value))))  
lambda_subcutis         = Constant(mesh, ScalarType(E_subcutis.value*nu_subcutis.value/((1+nu_subcutis.value)*(1-2*nu_subcutis.value))))  
mu_cutis                = Constant(mesh, ScalarType(E_cutis.value/(2*(1+nu_cutis.value))))  
mu_subcutis             = Constant(mesh, ScalarType(E_subcutis.value/(2*(1+nu_subcutis.value))))  
# 
# Map lambda
lambda_m                               = Function(DG0_space)
lambda_m.x.array[hypoderm_indices]     = np.full_like(hypoderm_indices, lambda_subcutis.value, dtype=ScalarType)
lambda_m.x.array[epiderm_indices]      = np.full_like(epiderm_indices, lambda_cutis.value, dtype=ScalarType)
lambda_m.x.scatter_forward()
# 
# Map mu
mu                               = Function(DG0_space)
mu.x.array[hypoderm_indices]     = np.full_like(hypoderm_indices, mu_subcutis.value, dtype=ScalarType)
mu.x.array[epiderm_indices]      = np.full_like(epiderm_indices, mu_cutis.value, dtype=ScalarType)
mu.x.scatter_forward()
#
# 
#------------------------------------------------------------#
#               Functions and expressions                    #
#------------------------------------------------------------#
# 
# Integral functions
# Specify the desired quadrature degree
q_deg = 4
dx    = Measure('dx', metadata={"quadrature_degree":q_deg}, subdomain_data=cell_tag, domain=mesh)
ds    = Measure("ds", domain=mesh, subdomain_data=facet_tag)
# Time discretisation
dt    = Constant(mesh, ScalarType(time_list[1]-time_list[0]))
# 
# Solution
# speed / delta pressure 
X0    = Function(MS)
# displacement / pressure
Xn    = Function(MS)
# 
du_update                = Function(updated_mesh_space)
displacement_export      = Function(CG1v_space)
displacement_export.name = "Displacement"
# Identify the unknowns from the function speed and delta pressure
# du = speed ; dp = pressure variation
du,dp    = split(X0)
u_n,p_n  = split(Xn)
# 
# Mapping in the Mixed Space
Un_, Un_to_MS = MS.sub(0).collapse()
Pn_, Pn_to_MS = MS.sub(1).collapse()
Initial_Pn_   = Function(Pn_)
# 
# Post-processing of the total stress
sigma_n      = Function(tensor_space) 
sigma_n.name = "EffectiveStress"
# 
# Post processing functions
velocity       = Function(CG1v_space)
velocity.name  = "FluidVelocity"
sigma_tot      = Function(tensor_space)
sigma_tot.name = "TotalStress"
react          = Function(CG1_space)
react.name     = "RF"
# Deformation of du (max of abs)
deformation = Function(tensor_space) 
deformation.name = "Max_def"
# Compute the effective stress using the stress rate. This formula is ok in elasticity simga_J = sigma_n+(dsigmap_n - sigma_n w + w sigma_n)*dt
# sigma_expr = sigma_expr        = Expression(Cauchy(mesh,Xn.sub(0)+X0.sub(0)*dt,lambda_m,mu)		
# 													+ (dot(spin_tensor(X0.sub(0)),sigma_n)-dot(sigma_n,spin_tensor(X0.sub(0))))*dt,
# 											tensor_space.element.interpolation_points())
# 
sigma_expr = sigma_expr        = Expression(Cauchy(mesh,Xn.sub(0)+X0.sub(0)*dt,lambda_m,mu),
											tensor_space.element.interpolation_points())
# 
# Compute the reaction Force
N                 = Constant(mesh, np.asarray(((1.0, 0.0,0.0),(0.0, 0.0,0.0),(0.0, 0.0,0.0))))
RF_expr           = form(inner(Cauchy(mesh,Xn.sub(0),lambda_m,mu)-beta*Xn.sub(1)*Identity(mesh.geometry.dim),N)*ds(1))
# 
# Compute the solid deformation increment
deformation_du_expr = Expression(sym(grad(X0.sub(0)*dt)), tensor_space.element.interpolation_points())
# Compute the total stress
sigma_tot_expr    = Expression(Cauchy(mesh,Xn.sub(0),lambda_m,mu)-beta*Xn.sub(1)*Identity(mesh.geometry.dim), tensor_space.element.interpolation_points())
# 
# Update of the porosity
poro_expr         = Expression(porosity_n + (1-porosity_n)*div((X0.sub(0))*dt), DG0_space.element.interpolation_points())
# 
# Update the fluid velocity (-k/(mu epsilon) grad(p_n) + v^s)
darcy_speed       = -(permeability/(mu*porosity))*grad(Xn.sub(1))+X0.sub(0)
velocity_expr     = Expression(darcy_speed, CG1v_space.element.interpolation_points())
# 
displacement_expr = Expression(Xn.sub(0),CG1v_space.element.interpolation_points())
# 
# Open file for export
xdmf = XDMFFile(mesh.comm, "Result_HE.xdmf", "w")
xdmf.write_mesh(mesh)
#------------------------------------------------------------#
#            Initial and Boundary Conditions                 #
#------------------------------------------------------------#
# 
# Initial IF Pressure [Pa]
pIF0 = 700
# 
with Initial_Pn_.vector.localForm() as initial_local:
	initial_local.set(ScalarType(pIF0)) 
Xn.x.array[Pn_to_MS] = Initial_Pn_.x.array[:]
Xn.x.scatter_forward()
# 
# Dirichlet BCs
# 
imp_speed = Constant(mesh,ScalarType(0.))
uprint = 0
# 	Semi-infinite boundaries
# right_patch_marker, right_marker, left_marker, front_marker, back_marker, bottom_marker, top_marker, u_right_marker 
# 1                 ,2            ,3           ,4            ,5           ,6             ,7          ,8
fdim   = mesh.topology.dim - 1
bcs    = []
# 1: u_x=u_imp ; u_y = u_z = 0
facets = facet_tag.find(1)
dofs   = locate_dofs_topological(MS.sub(0).sub(0), fdim, facets)
bcs.append(dirichletbc(imp_speed, dofs, MS.sub(0).sub(0)))
dofs   = locate_dofs_topological(MS.sub(0).sub(1), fdim, facets)
bcs.append(dirichletbc(ScalarType(0), dofs, MS.sub(0).sub(1)))
dofs   = locate_dofs_topological(MS.sub(0).sub(2), fdim, facets)
bcs.append(dirichletbc(ScalarType(0), dofs, MS.sub(0).sub(2)))
# 2: ux=0
facets = facet_tag.find(3)
dofs   = locate_dofs_topological(MS.sub(0).sub(0), fdim, facets)
bcs.append(dirichletbc(ScalarType(0), dofs, MS.sub(0).sub(0)))
# 6: uy=0 symmetry
facets = facet_tag.find(5)
dofs   = locate_dofs_topological(MS.sub(0).sub(1), fdim, facets)
bcs.append(dirichletbc(ScalarType(0), dofs, MS.sub(0).sub(1)))
# 7: uz=0
facets = facet_tag.find(6)
dofs   = locate_dofs_topological(MS.sub(0).sub(2), fdim, facets)
bcs.append(dirichletbc(ScalarType(0), dofs, MS.sub(0).sub(2)))
# 8: u_x = u_imp ; u_y = u_z = 0
facets = facet_tag.find(8)
dofs   = locate_dofs_topological(MS.sub(0).sub(0), fdim, facets)
bcs.append(dirichletbc(imp_speed, dofs, MS.sub(0).sub(0)))
dofs   = locate_dofs_topological(MS.sub(0).sub(1), fdim, facets)
bcs.append(dirichletbc(ScalarType(0), dofs, MS.sub(0).sub(1)))
dofs   = locate_dofs_topological(MS.sub(0).sub(2), fdim, facets)
bcs.append(dirichletbc(ScalarType(0), dofs, MS.sub(0).sub(2)))
# 
# 
#------------------------------------------------------------#
#                     Variationnal form                      #
#------------------------------------------------------------#
#
# Define the test functions
v,q = TestFunctions(MS)
# 
F   = (nabla_div(du)*q*dx + (permeability/viscosity)*dot(grad(p_n+dp),grad(q))*dx  + ( S/dt )*(dp)*q*dx)
# 
# F  += 1/(dt)*((inner(grad(v),(Cauchy(mesh,u_n+du*dt,lambda_m,mu)-Cauchy(mesh,u_n,lambda_m,mu))) * dx
# 					- beta * (dp) * nabla_div(v)*dx))
F  += ((inner(grad(v),(Cauchy(mesh,u_n+du*dt,lambda_m,mu))) * dx
					- beta * (p_n+dp) * nabla_div(v)*dx))
# 
#------------------------------------------------------------#
#                           Solver                           #
#------------------------------------------------------------#
# 
# Non linear problem definition
dX0                          = TrialFunction(MS)
J                            = derivative(F, X0, dX0)
Problem                      = NonlinearProblem(F, X0, bcs = bcs, J = J)
# set up the non-linear solver
solver                       = NewtonSolver(mesh.comm, Problem)
# Absolute tolerance
solver.atol                  = 1e-9
# relative tolerance
solver.rtol                  = 1e-10
# Convergence criterion
solver.convergence_criterion = "incremental"
# Maximum iterations
solver.max_it                = 30
# 
log_newton=True
if log_newton:
	from dolfinx import log
	log.set_log_level(log.LogLevel.INFO)
#------------------------------------------------------------#
#                         Computation                        #
#------------------------------------------------------------#
# 
Jcost        = 0
list_t       = []
list_t_du       = []
list_react   = []
verif_du = []
mean_du = []
std_du=[]
imposed_disp = 0
number_rmse  = 0
for n in range(num_steps-1):
	dt.value = time_list[n+1]-time_list[n]
	t += dt.value
	list_t.append(t)
	# Beware we are computing the speed du
	imp_speed.value = (real_u_m[n+1]-real_u_m[n])/dt.value
	imposed_disp    = (real_u_m[n+1]-real_u_m[n])
	uprint+=imposed_disp
	if MPI.COMM_WORLD.rank == 0:
		print(f"Time step {n+1}/{num_steps-1}, dt: {dt.value}, Imposed displacement {uprint} m, time = {time_list[n+1]}")
	try:
		num_its, converged = solver.solve(X0)
	except:
		if MPI.COMM_WORLD.rank == 0:
			print("*************") 
			print("Solver failed")
			print("*************")
			break 
	X0.x.scatter_forward()
	deformation.interpolate(deformation_du_expr)
	deformation.x.scatter_forward()
	verif_du.append(100*max(abs(deformation.x.array[:])))
	mean_du.append(100*np.mean(abs(deformation.x.array[:])))
	std_du.append(100*np.std(abs(deformation.x.array[:])))
	list_t_du.append(t)
	if MPI.COMM_WORLD.rank == 0:
		print(f"100*max(abs(epsilon du*dt)) = {verif_du[-1]} %")
		print(f"100*mean(abs(epsilon du*dt)) = {mean_du[-1]} %")
		print(f"100*std(abs(epsilon du*dt)) = {std_du[-1]} %")
	# Update the effective stress
	sigma_n.interpolate(sigma_expr) 
	sigma_n.x.scatter_forward()
	# Update Value u_n & p_n
	Xn.x.array[Un_to_MS] += X0.x.array[Un_to_MS]*dt.value
	Xn.x.array[Pn_to_MS] += X0.x.array[Pn_to_MS]
	Xn.x.scatter_forward()
	# Update the effective stress
	sigma_tot.interpolate(sigma_tot_expr) 
	sigma_tot.x.scatter_forward()
	# Update porosity
	porosity.interpolate(poro_expr) 
	porosity.x.scatter_forward()
	# Update porosity at previous time step
	porosity_n.x.array[:]=porosity.x.array[:]
	porosity_n.x.scatter_forward()
	# Displacement to CG1 for export
	displacement_export.interpolate(displacement_expr)
	displacement_export.x.scatter_forward()
	# Compute speed
	velocity.interpolate(velocity_expr)
	velocity.x.scatter_forward()
	# 
	react_local = assemble_scalar(RF_expr)
	react = (mesh.comm.allreduce(react_local, op=MPI.SUM))
	list_react.append(react)
	# Export the results
	__u, __p = Xn.split()
	__p.name = "Pressure"
	# 
	xdmf.write_function(displacement_export,t)
	xdmf.write_function(__p,t)
	xdmf.write_function(sigma_n,t)
	xdmf.write_function(sigma_tot,t)
	xdmf.write_function(porosity,t)
	xdmf.write_function(velocity,t)
	# 
	# mesh update
	du_update.interpolate(X0.sub(0))
	mesh.geometry.x[:,:mesh.geometry.dim] += du_update.x.array.reshape((-1, mesh.geometry.dim))*dt.value
	#
	# RMSE eval
	if ((t >= 18.8) and (t<=75)) or ((t>=97.8) and (t<=145)):
		Jcost       += (RF_exp[n+1]-react)**2
		number_rmse += 1 
	# 
# 
xdmf.close()
# 
# 
###################################################
###################################################
################ Post-Processing ##################
###################################################
###################################################
# 
# 
if MPI.COMM_WORLD.rank == 0:
	# Export data to CSV
	export_to_csv(np.transpose([list_t,list_react,verif_du]), 'RF_HE.csv')
	# 
	###################################################
	###################################################
	################ Plots ############################
	###################################################
	###################################################
	# 
	import matplotlib.pyplot as plt
	# 
	plt.rcParams.update({'font.size': 15})
	plt.rcParams.update({'legend.loc':'upper right'})
	# 
	fig1, ax1 = plt.subplots()
	ax1.plot(time_second,force_newton,linestyle='-',linewidth=2,color='powderblue')
	ax1.set_xlabel('time (s)')
	ax1.set_ylabel('Reaction Force (N)')
	fig1.tight_layout()
	fig1.savefig('Experimental_BHE.jpg')
	# 
	fig2, ax2 = plt.subplots()
	ax2.plot(list_t,list_react,linestyle='-',linewidth=2,color='powderblue')
	ax2.set_ylabel('Reaction Force (N)')
	ax2.set_xlabel('time (s)')
	fig2.tight_layout()
	fig2.savefig('Model_BHE.jpg')
	# Superimposed
	fig2, ax2 = plt.subplots()
	ax2.plot(time_second,force_newton,linestyle='-',linewidth=2,color='lightgreen',label='Exp')
	ax2.plot(list_t,list_react,linestyle=':',linewidth=2,color='olivedrab',label='theo')
	ax2.set_ylabel('Reaction Force (N)')
	ax2.set_xlabel('time (s)')
	ax2.legend()
	fig2.tight_layout()
	fig2.savefig('Model_B_2_8mmHE.jpg')
	fig2, ax2 = plt.subplots()
	ax2.plot(list_t_du,verif_du,linestyle='-',linewidth=1.5,color='olivedrab',label='max')
	ax2.plot(list_t_du,mean_du,linestyle='-.',linewidth=1.5,label='mean')
	ax2.plot(list_t_du,std_du,linestyle=':',linewidth=1.5,label='std')
	ax2.set_ylabel('100*op(abs(epsilon(du))) %')
	ax2.set_xlabel('time (s)')
	ax2.legend()
	fig2.tight_layout()
	fig2.savefig('approx_small_disp.jpg')
# 
# Evaluate final time
end_t = time.time()
if MPI.COMM_WORLD.rank == 0:	
		try:	
			print("Jcost [N] = ",(math.sqrt((1/number_rmse)*float(Jcost))))
		except:
			print("No points were included in RMSE")
# 
# Evaluate final time
end_t = time.time()
t_hours = int((end_t-begin_t)//3600)
tmin = int(((end_t-begin_t)%3600)//60)
tsec = int(((end_t-begin_t)%3600)%60)
if MPI.COMM_WORLD.rank == 0:
	print(f"FComputation: {num_steps} iterations in {t_hours} h {tmin} min {tsec} sec")
exit()