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
time_second, _ , displacement_milli_meter, force_newton = read_excel_sheet(file_path, sheet_name)
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
dt_objective  = 0.50
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
Nzeros = 19
RF_exp[:Nzeros]=np.zeros(Nzeros) 
# 
# Identify the number of steps for the computation
# num_steps = len(real_u_m)
num_steps = time_list.index(145)
##############################################################
##############################################################
########################## Computation #######################
##############################################################
##############################################################
# 
# 
def problem_opti(param, gradient):
	"""
	Function which computes the cost function for a set of parameters
	"""
	# initialize cost function
	Jcost = 0
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
	#
	# Identify indices of the cells for each region for material definition
	hypoderm_indices = [x for x in cell_tag.indices if (cell_tag.values[x] == 200)]
	epiderm_indices  = [x for x in cell_tag.indices if (cell_tag.values[x] == 100)]
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
	E_cutis               = Constant(mesh, ScalarType(param[0]*150e3)) 
	E_subcutis            = Constant(mesh, ScalarType(param[1]*50e3))  
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
	permeability_cutis    = Constant(mesh, ScalarType(param[2]*1e-14)) 
	permeability_subcutis = Constant(mesh, ScalarType(param[3]*1e-13)) 
	# Fluid bulk modulus [Pa]
	Kf                    = Constant(mesh, ScalarType(2.2e9))
	# Solid bulk modulus [Pa]
	Ks                    = Constant(mesh, ScalarType(1e10))
	# Biot Coefficient [-]
	beta                  = Constant(mesh, ScalarType(1))
	# 
	if MPI.COMM_WORLD.rank == 0:
		print(f"Auxiliary parameter for cutis : E={param[0]}, k={param[2]} ; Auxiliary parameter for subcutis E= {param[1]}, k= {param[3]}")
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
	# 
	# Compute the reaction Force
	N                 = Constant(mesh, np.asarray(((1.0, 0.0,0.0),(0.0, 0.0,0.0),(0.0, 0.0,0.0))))
	RF_expr           = form(inner(Cauchy(mesh,Xn.sub(0),lambda_m,mu)-beta*Xn.sub(1)*Identity(mesh.geometry.dim),N)*ds(1))
	# 
	# Update of the porosity (ask ref to S Urcun)
	poro_expr         = Expression(porosity_n + (1-porosity_n)*div((X0.sub(0))*dt), DG0_space.element.interpolation_points())
	# 
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
	solver.atol                  = 5e-10
	# relative tolerance
	solver.rtol                  = 1e-11
	# Convergence criterion
	solver.convergence_criterion = "incremental"
	# Maximum iterations
	solver.max_it                = 10
	# 
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
	print('Begin solve')
	for n in range(num_steps-1):
		dt.value = time_list[n+1]-time_list[n]
		t += dt.value
		list_t.append(t)
		# Beware we are computing the speed du
		imp_speed.value = (real_u_m[n+1]-real_u_m[n])/dt.value
		imposed_disp    = (real_u_m[n+1]-real_u_m[n])
		uprint+=imposed_disp
		try:
			num_its, converged = solver.solve(X0)
		except:
			if MPI.COMM_WORLD.rank == 0:
				print("*************") 
				print("Solver failed")
				print("*************")
			Jcost = 1e6 
			return float(Jcost)
		X0.x.scatter_forward()
		# Update Value u_n & p_n
		Xn.x.array[Un_to_MS] += X0.x.array[Un_to_MS]*dt.value
		Xn.x.array[Pn_to_MS] += X0.x.array[Pn_to_MS]
		Xn.x.scatter_forward()
		# Update porosity
		porosity.interpolate(poro_expr) 
		porosity.x.scatter_forward()
		# Update porosity at previous time step
		porosity_n.x.array[:]=porosity.x.array[:]
		porosity_n.x.scatter_forward()
		# 
		react_local = assemble_scalar(RF_expr)
		react = (mesh.comm.allreduce(react_local, op=MPI.SUM))
		list_react.append(react)
		# 
		# mesh update
		du_update.interpolate(X0.sub(0))
		mesh.geometry.x[:,:mesh.geometry.dim] += du_update.x.array.reshape((-1, mesh.geometry.dim))*dt.value
		#
		# RMSE eval
		if ((t >= 18.8) and (t<=75)) or ((t>=97.8) and (t<=145)):
			Jcost       += (RF_exp[n+1]-react)**2
			number_rmse += 1 
	print('End solve')
		# 
	if MPI.COMM_WORLD.rank == 0:		
		print("Jcost (RMSE unit) = ",(math.sqrt((1/number_rmse)*float(Jcost))),"Jcost_max (RMSE unit) = ",(math.sqrt((np.max(RF_exp)-np.max(react))**2)))
		import matplotlib.pyplot as plt
		# 
		plt.rcParams.update({'font.size': 15})
		plt.rcParams.update({'legend.loc':'upper right'})
		# Superimposed
		fig2, ax2 = plt.subplots()
		ax2.plot(time_list[:num_steps],RF_exp[:num_steps],linestyle='-',linewidth=2,color='lightgreen',label='Exp')
		ax2.plot(list_t,list_react,linestyle=':',linewidth=2,color='olivedrab',label='model')
		ax2.set_ylabel('Reaction Force (N)')
		ax2.set_xlabel('time (s)')
		ax2.legend()
		fig2.tight_layout()
		fig2.savefig('Opti_current_output.jpg')
		plt.close('all')
	# 
	return math.sqrt((1/number_rmse)*float(Jcost))
# 
# 
# 
# 
if __name__ == "__main__":
	# 
	import time
	import os
	import glob
	import shutil
	import nlopt
	# 
	## Create the output directories and files
	# Output directory
	directory = "Results_NLOPT_GN_CSR"
	# Parent Directory path
	parent_dir = "./"
	# Path
	path       = os.path.join(parent_dir, directory)
	try:
		os.mkdir(path)
	except:
		pass
	# 
	parameter_file = "./Results_NLOPT_GN_CSR/optimized parameters.txt"
	initial_p_file = "./Results_NLOPT_GN_CSR/initial_parameters.txt"
	others_file    = "./Results_NLOPT_GN_CSR/readme.txt"
	# 
	myopt = open(parameter_file, "w")
	myopt.close()
	myinit = open(initial_p_file, "w")
	myinit.close()
	myrdm = open(others_file, "w")
	myrdm.close()
	# E1 E2 K1 K2
	# x0 = [1., 1., 1., 1.]
	## Initial parameter
	x0=[1, 1, 4, 3]
	# lower bound
	lb = [0.1, 0.1, 1e-1, 1e-2]
	# upper bound
	ub = [1e2, 1e2, 1e3, 1e2]
	maxeval = 250
	NN = len(x0)
	# # Set time counter
	begin_t = time.time()
	# opt = nlopt.opt(nlopt.GN_ISRES , NN)
	opt = nlopt.opt(nlopt.GN_CRS2_LM ,NN)
	# opt = nlopt.opt(nlopt._DIRECT ,NN)
	opt.set_min_objective(problem_opti)
	opt.set_population(3*(NN+1))
	# opt.add_inequality_constraint(lambda x,grad: myconstraint(x,grad,2,0), 1e-8)
	# opt.add_inequality_constraint(lambda x,grad: myconstraint(x,grad,-1,1), 1e-8)
	# opt.set_stopval(stopval)
	opt.set_xtol_rel(1e-6)
	opt.set_ftol_rel(1e-8)
	# opt.set_ftol_abs(1e-4)
	opt.set_lower_bounds(lb)
	opt.set_upper_bounds(ub)
	opt.set_maxeval(maxeval)
	x = opt.optimize(x0)
	minf = opt.last_optimum_value()
	print("optimum at ", x[0], x[1], x[2], x[3])
	print("minimum value = ", minf)
	print("result code = ", opt.last_optimize_result())
	# 
	# 
	# Evaluate final time
	end_t = time.time()
	t_hours = int((end_t-begin_t)//3600)
	tmin = int(((end_t-begin_t)%3600)//60)
	tsec = int(((end_t-begin_t)%3600)%60)
	# 
	print("solution: ", x[0], x[1], x[2], x[3])
	print(f"Optimisation operated in {t_hours} h {tmin} min {tsec} sec")
	print(f"Optimisation result code : {opt.last_optimize_result()}")
	print(f"Value of objective function: {minf}")
	print(f"Values in the parameter file corresponds to the alternate parameter value. (E/Eref, permeability/permeabilityref)")
	myopt = open(parameter_file, "a")
	myopt.write(f"{x[0]} {x[1]} {x[2]} {x[3]} \n")
	myopt.close()
	myinit = open(initial_p_file, "a")
	myinit.write(f"{x0[0]} {x0[1]} {x0[2]} {x0[3]}\n")
	myinit.close()
	myrdm = open(others_file, "a")
	myrdm.write(f"Optimisation operated in {t_hours} h {tmin} min {tsec} sec \n")
	myrdm.write(f"Optimisation result code : {opt.last_optimize_result()} \n")
	myrdm.write(f"Value of objective function: {minf} \n")
	myrdm.write(f"Values in the parameter file corresponds to the alternate parameter value. (E/Eref, permeability/permeabilityref)\n")
	myrdm.write(f"\n")
	myrdm.write(f"\n")
	myrdm.close()
# 
# Successful termination (positive return values)
# NLOPT_SUCCESS = 1
# Generic success return value.
# NLOPT_STOPVAL_REACHED = 2
# Optimization stopped because stopval (above) was reached.
# NLOPT_FTOL_REACHED = 3
# Optimization stopped because ftol_rel or ftol_abs (above) was reached.
# NLOPT_XTOL_REACHED = 4
# Optimization stopped because xtol_rel or xtol_abs (above) was reached.
# NLOPT_MAXEVAL_REACHED = 5
# Optimization stopped because maxeval (above) was reached.
# NLOPT_MAXTIME_REACHED = 6
# Optimization stopped because maxtime (above) was reached.
exit()