# Thomas Lavigne
# 18/07/2023
# SKIN
import numpy as np
import time
import csv
from petsc4py          import PETSc
import dolfinx
from dolfinx           import nls
from dolfinx.io        import XDMFFile, gmshio
from dolfinx.mesh      import CellType, create_box, refine, locate_entities_boundary, locate_entities, meshtags
from dolfinx.fem       import (Constant, dirichletbc, Function, FunctionSpace, Expression, locate_dofs_topological, form, assemble_scalar)
from dolfinx.fem.petsc import NonlinearProblem
#from dolfinx.geometry import BoundingBoxTree, compute_collisions, compute_colliding_cells
from petsc4py.PETSc    import ScalarType
from mpi4py            import MPI
from ufl               import (variable, skew, as_tensor, as_vector, as_matrix, TensorElement,SpatialCoordinate,FacetNormal, Identity, Measure, TestFunctions, TrialFunction, VectorElement, FiniteElement, dot, dx, inner, grad, nabla_div, div, sym, MixedElement, 
									derivative, split,sqrt)
from scipy.optimize    import minimize
import matplotlib.pyplot as plt
import math
# 
###################################################
###################################################
#################### Functions ####################
###################################################
###################################################
# 
# 
def teff_NH_UJ1(u,lambda_m,mu):
	"""
	Compute the stress from a compressible neo-Hookean formulation: W = (mu / 2) * (Ic - tr(Identity(len(u)))) - mu * ln(J) + (lambda_m / 2) * (ln(J))^2
	Inputs:
	- lambda_m, mu : Lame_Coefficients
	- u : displacement 
	Outputs: 
	- diff(W, F): First Piola Kirchoff stress tensor
	"""
	from ufl import variable, Identity, grad, det, tr, ln, diff
	## Deformation gradient
	F = variable(Identity(mesh.geometry.dim) + grad(u))
	J  = variable(det(F))
	## Right Cauchy-Green tensor
	C = variable(F.T * F)
	##Invariants of deformation tensors
	Ic = variable(tr(C))
	W = (mu / 2) * (Ic - tr(Identity(mesh.geometry.dim))) - mu * ln(J) + (lambda_m / 2) * (ln(J))**2
	return diff(W, F)
#
# 
def read_excel_sheet(file_path, sheet_name):
	import pandas as pd
	# Read the specified sheet from the Excel file
	df = pd.read_excel(file_path, sheet_name=sheet_name)
	
	# Extract the first four columns as separate lists
	column1 = df.iloc[:, 0].tolist()
	column2 = df.iloc[:, 1].tolist()
	column3 = df.iloc[:, 2].tolist()
	column4 = df.iloc[:, 3].tolist()
	
	# Return the four lists
	return column1, column2, column3, column4
#
def keep_every_n_elements_A_Sain(lst):
	result = []
	for i in range(len(lst)):
		n = 50
		if i % n == 0:
			result.append(lst[i])
	return result
# 
def calculer_moyenne_g(nombre_elements,data):
	moyenne_gauche = []
	for i in range(len(data)):
		if i < nombre_elements:
			gauche = data[:i]
		else:
			gauche = data[i - nombre_elements:i]
		if len(gauche)==0:
			moyenne_gauche.append(data[i])
		else:
			moyenne_gauche.append(calculer_moyenne(gauche))
	return moyenne_gauche

def calculer_moyenne_d(nombre_elements,data):
	moyenne_droite = []
	for i in range(len(data)):
		droite = data[i + 1:i + nombre_elements + 1]
		if len(droite)==0:
			moyenne_droite.append(data[i])
		else:
			moyenne_droite.append(calculer_moyenne(droite))
	return moyenne_droite

def calculer_moyenne_c(nombre_elements,data):
	moyenne_centree = []
	for i in range(len(data)):
		centree = data[max(0, i - nombre_elements // 2):i]
		if i + nombre_elements // 2 < len(data):
			centree += data[i + 1:i + nombre_elements // 2 + 1]
		else:
			centree += data[i + 1:]
		if len(centree)==0:
			moyenne_centree.append(data[i])
		else:
			moyenne_centree.append(calculer_moyenne(centree))
	return moyenne_centree
# 
def calculer_moyenne(liste):
    if len(liste) == 0:
        return 0
    return np.mean(liste)
# 
# 
###################################################
######### Import the experimental data ############
###################################################
# 
# Specify the file path and sheet name
file_path = '../Experimental_Data/Healthy_skin.xlsx'
sheet_name = 'Test B - Sain'
# 
# Read the specified sheet and get the four columns as lists
column1_list, column2_list, column3_list, column4_list = read_excel_sheet(file_path, sheet_name)
for ii in range(len(column1_list)):
	if column1_list[ii]<=30:
		n0=ii
	elif column1_list[ii]<=75:
		n1=ii
	elif column1_list[ii]<=90:
		n2=ii
	elif column1_list[ii]<=105:
		n3=ii
	elif column1_list[ii]<=145:
		n4=ii

column4_list[:n0]=calculer_moyenne_g(50,column4_list[:n0])
column4_list[n0:n1]=calculer_moyenne_c(1000,column4_list[n0:n1])
column4_list[n1:n2]=calculer_moyenne_d(50,column4_list[n1:n2])
column4_list[n2:n3]=calculer_moyenne_g(50,column4_list[n2:n3])
column4_list[n3:n4]=calculer_moyenne_c(1000,column4_list[n3:n4])
# 
# 
time_list=keep_every_n_elements_A_Sain(column1_list)
theorical_u_=keep_every_n_elements_A_Sain(column2_list)
real_u_ = keep_every_n_elements_A_Sain(column3_list)
RF_exp=keep_every_n_elements_A_Sain(column4_list)
# correction pre stress
Nzeros = 19
RF_exp[:Nzeros]=np.zeros(Nzeros) 
###################################################
################ Indentation ######################
###################################################	
# 
# Scaling mm to m and half for symmetry
theorical_u = [1e-3*elem/2 for elem in theorical_u_]
real_u = [1e-3*elem/2 for elem in real_u_]
# real_u = np.linspace(0,1e-4,3)
# time_list = np.linspace(0,0.05,3)
num_steps = len(real_u)
# for ii in range(len(time_list)):
# 	if time_list[ii]>=145:
# 		num_steps=ii
# 		time_list=time_list[:num_steps]
# 		theorical_u = theorical_u[:num_steps]
# 		real_u = real_u[:num_steps]
# 		RF_exp = RF_exp[:num_steps]
# 		break	
# 
# Set time counter
begin_t = time.time()
# 
# ###################################################
# ################ User defined #####################
# ###################################################	
# # 
# 
## Create the domain / mesh
filename = "../Mesh_GMSH/Mesh_file/Symmetric_skin_XY_RF.msh"
# right_patch_marker, right_marker, left_marker, front_marker, back_marker, bottom_marker, top_marker, u_right_marker = 1,2,3,4,5,6,7,8
mesh, cell_tag, facet_tag = gmshio.read_from_msh(filename, MPI.COMM_WORLD, 0, gdim=3)
# Deform the mesh
R = 48e-3
zx,zy,zz = 1e-16, 1e-16, 1e-16# 
center_y = zy
# 
# def du_(x):
# 	return -R*np.ones_like(x[0])+np.sqrt((R*np.ones_like(x[0]))**2-(x[1]-center_y*np.ones_like(x[0]))**2)
def du_y(x):
	theta = x[1]/(R+x[2])*np.ones_like(x[0])
	return -x[1]+(R+x[2])*np.sin(theta)
def du_z(x):
	theta = x[1]/(R+x[2])*np.ones_like(x[0])
	return -x[2]+(R+x[2])*np.cos(theta)
# 
updated_mesh_space    = FunctionSpace(mesh, mesh.ufl_domain().ufl_coordinate_element())
# Evaluation of the displacement to update the mesh (updated Lagrangian)
d_u_ = Function(updated_mesh_space)
d_u_.interpolate(lambda x: np.stack((np.zeros_like(x[0]), du_y(x), du_z(x))))
d_u_.x.scatter_forward()
mesh.geometry.x[:,:mesh.geometry.dim] += d_u_.x.array.reshape((-1, mesh.geometry.dim))

# Save output for debug
with XDMFFile(MPI.COMM_WORLD, "mesh_tag.xdmf", "w") as xdmftag:
		xdmftag.write_mesh(mesh)
		xdmftag.write_meshtags(facet_tag)
		xdmftag.write_meshtags(cell_tag)
		# xdmftag.write_meshtags(facet_tag, mesh.geometry)
		# xdmftag.write_meshtags(cell_tag, mesh.geometry)


# Identify indices of the cells for each region for material definition
hypoderm_indices = [x for x in cell_tag.indices if (cell_tag.values[x] == 200)]
epiderm_indices = [x for x in cell_tag.indices if (cell_tag.values[x] == 100)]
# Debug Check
try :
	assert(len(cell_tag.indices) == len(epiderm_indices)+len(hypoderm_indices))
	if MPI.COMM_WORLD.rank == 0:
		print("All cell tags have been attributed")
except:
	if MPI.COMM_WORLD.rank == 0:
		print("*************") 
		print("Forgotten tags => material badly defined")
		print("*************") 
		exit()

# 
# Solid scaffold
E1            = Constant(mesh, ScalarType(1500e3)) 
E2            = Constant(mesh, ScalarType(100e3))  
nu1           = Constant(mesh, ScalarType(0.45))
nu2           = Constant(mesh, ScalarType(0.3))
rhos         = Constant(mesh, ScalarType(1))
lambda1			= Constant(mesh, ScalarType(E1.value*nu1.value/((1+nu1.value)*(1-2*nu1.value))))  
lambda2       = Constant(mesh, ScalarType(E2.value*nu2.value/((1+nu2.value)*(1-2*nu2.value))))  
mu1          = Constant(mesh, ScalarType(E1.value/(2*(1+nu1.value))))  
mu2          = Constant(mesh, ScalarType(E2.value/(2*(1+nu2.value))))  
# Porous material
viscosity    = Constant(mesh, ScalarType(5e-3))   
porosity1     = Constant(mesh, ScalarType(0.2))
porosity2     = Constant(mesh, ScalarType(0.4))
permeability1 = Constant(mesh, ScalarType(1e-12)) 
permeability2 = Constant(mesh, ScalarType(1e-14)) 
Kf           = Constant(mesh, ScalarType(2.2e9))
Ks           = Constant(mesh, ScalarType(1e10))
rhol         = Constant(mesh, ScalarType(1))
beta         = Constant(mesh, ScalarType(1))
# 
# Repartition of the material parameters within the domain
parameter_space = FunctionSpace(mesh, ("DG", 0))


# Poisson Ratio
nu = Function(parameter_space)
nu.x.array[hypoderm_indices] = np.full_like(hypoderm_indices, nu2.value, dtype=ScalarType)
nu.x.array[epiderm_indices] = np.full_like(epiderm_indices, nu1.value, dtype=ScalarType)
nu.x.scatter_forward()
# 
porosity = Function(parameter_space)
porosity.x.array[hypoderm_indices]     = np.full_like(hypoderm_indices, porosity2.value, dtype=ScalarType)
porosity.x.array[epiderm_indices]      = np.full_like(epiderm_indices, porosity1.value, dtype=ScalarType)
porosity.x.scatter_forward()
# Storativity coefficient
S            = (porosity/Kf)+(1-porosity)/Ks

###################################################
########### Functionnal to minimize ###############
###################################################
def eval_opti(xpar):
	# Initialize the counter and the cost function values.
	Jcost = variable(0.0)
	# xpar Young epirderm, Young Hypoderm ; Perm epiderm, Perm hypoderm
	# 
	# SOLID SCAFFOLD
	# Young Modulus
	E = Function(parameter_space)
	E.x.array[hypoderm_indices] = np.full_like(hypoderm_indices, xpar[1]*E2.value, dtype=ScalarType)
	E.x.array[epiderm_indices] = np.full_like(epiderm_indices, xpar[0]*E1.value, dtype=ScalarType)
	E.x.scatter_forward()
	# Lame coefficients
	lambda1			= Constant(mesh, ScalarType(xpar[0]*E1.value*nu1.value/((1+nu1.value)*(1-2*nu1.value))))  
	lambda2       = Constant(mesh, ScalarType(xpar[1]*E2.value*nu2.value/((1+nu2.value)*(1-2*nu2.value))))  
	mu1          = Constant(mesh, ScalarType(xpar[0]*E1.value/(2*(1+nu1.value))))  
	mu2          = Constant(mesh, ScalarType(xpar[1]*E2.value/(2*(1+nu2.value))))  
	# 
	lambda_m = Function(parameter_space)
	lambda_m.x.array[hypoderm_indices]     = np.full_like(hypoderm_indices, lambda2.value, dtype=ScalarType)
	lambda_m.x.array[epiderm_indices]      = np.full_like(epiderm_indices, lambda1.value, dtype=ScalarType)
	lambda_m.x.scatter_forward()
	# 
	mu = Function(parameter_space)
	mu.x.array[hypoderm_indices]     = np.full_like(hypoderm_indices, mu2.value, dtype=ScalarType)
	mu.x.array[epiderm_indices]      = np.full_like(epiderm_indices, mu1.value, dtype=ScalarType)
	mu.x.scatter_forward()
	# 
	# POROUS MEDIA
	permeability = Function(parameter_space)
	permeability.x.array[hypoderm_indices]     = np.full_like(hypoderm_indices, xpar[3]*permeability2.value, dtype=ScalarType)
	permeability.x.array[epiderm_indices]      = np.full_like(epiderm_indices, xpar[2]*permeability1.value, dtype=ScalarType)
	permeability.x.scatter_forward()
	if MPI.COMM_WORLD.rank == 0:
		print(f"Auxiliary parameter for E1={xpar[0]}, E2={xpar[1]} ; Auxiliary parameter for permeability1= {xpar[2]},permeability2= {xpar[3]}")
		print(f"E1={xpar[0]*E1.value}, E2={xpar[1]*E2.value}; permeability1 = {xpar[2]*permeability1.value}, permeability2 = {xpar[3]*permeability2.value}")
	# 
	###################################################
	########## Mixed Space of resolution ##############
	###################################################
	# 
	# Define Mixed Space (R2,R) -> (u,p)
	displacement_element  = VectorElement("CG", mesh.ufl_cell(), 2)
	pressure_element      = FiniteElement("CG", mesh.ufl_cell(), 1)
	updated_mesh_space    = FunctionSpace(mesh, mesh.ufl_domain().ufl_coordinate_element())
	displacement_space 	  = FunctionSpace(mesh, displacement_element)
	pressure_space 		  = FunctionSpace(mesh, pressure_element)
	MS                    = FunctionSpace(mesh, MixedElement([displacement_element,pressure_element]))
	# 
	# Create the initial and solution functions of space
	X0 = Function(MS)
	Xn = Function(MS)
	# # Evaluation of the displacement to update the mesh (updated Lagrangian)
	# Post-processing of the total stress
	sigma_tot_elem = TensorElement("CG", mesh.ufl_cell(), degree=1, shape=(3,3))
	sigma_tot_space = FunctionSpace(mesh, sigma_tot_elem)
	sigma_n = Function(sigma_tot_space) 
	# Post processing functions
	t_tots = Function(sigma_tot_space)
	V_RF = FunctionSpace(mesh, ("CG", 1))
	react = Function(V_RF)
	# 
	Pn_, Pn_to_MS = MS.sub(1).collapse()
	FPn_ = Function(Pn_)
	with FPn_.vector.localForm() as initial_local:
		initial_local.set(ScalarType(700)) 
	Xn.x.array[Pn_to_MS] = FPn_.x.array
	Xn.x.scatter_forward()
	###################################################
	######## Dirichlet boundary condition #############
	###################################################
	imp_disp = Constant(mesh,ScalarType(0.))
	# 	Semi-infinite boundaries
	# right_patch_marker, right_marker, left_marker, front_marker, back_marker, bottom_marker, top_marker, u_right_marker = 1,2,3,4,5,6,7,8
	# bcs    = []
	fdim = mesh.topology.dim - 1
	bcs=[]
	# 
	# 1: ux=imp
	facets = facet_tag.find(1)
	dofs   = locate_dofs_topological(MS.sub(0).sub(0), fdim, facets)
	bcs.append(dirichletbc(imp_disp, dofs, MS.sub(0).sub(0)))
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
	# 10: ux = uimp
	facets = facet_tag.find(8)
	dofs   = locate_dofs_topological(MS.sub(0).sub(0), fdim, facets)
	bcs.append(dirichletbc(imp_disp, dofs, MS.sub(0).sub(0)))
	dofs   = locate_dofs_topological(MS.sub(0).sub(1), fdim, facets)
	bcs.append(dirichletbc(ScalarType(0), dofs, MS.sub(0).sub(1)))
	dofs   = locate_dofs_topological(MS.sub(0).sub(2), fdim, facets)
	bcs.append(dirichletbc(ScalarType(0), dofs, MS.sub(0).sub(2)))
	# 
	# 
	###################################################
	############## Variationnal form ##################
	###################################################
	q_deg = 4# (Desired quadrature degree)
	dx = Measure('dx', metadata={"quadrature_degree":q_deg}, subdomain_data=cell_tag, domain=mesh)
	# dx = dx(metadata={"quadrature_degree":q_deg})
	ds = Measure("ds", domain=mesh, subdomain_data=facet_tag)
	dt = Constant(mesh, ScalarType(time_list[1]-time_list[0]))
	# Identify the unknowns from the function
	u,p    =split(X0)
	u_n,p_n=split(Xn)
	# Expressions
	sigma_expr = Expression(teff_NH_UJ1(u,lambda_m,mu), sigma_tot_space.element.interpolation_points())
	t_tot_expr = Expression(teff_NH_UJ1(u,lambda_m,mu)-beta*Xn.sub(1)*Identity(3), sigma_tot_space.element.interpolation_points())
	N = Constant(mesh, np.asarray(((1.0, 0.0,0.0),(0.0, 0.0,0.0),(0.0, 0.0,0.0))))
	RF_expr = form(inner(teff_NH_UJ1(Xn.sub(0),lambda_m,mu)-beta*Xn.sub(1)*Identity(3),N)*ds(1))
	# 
	# Set up the test functions & weak form
	v,q = TestFunctions(MS)
	F  = (1/dt*nabla_div(u-u_n)*q*dx + (permeability/viscosity)*dot(grad(p),grad(q))*dx  + ( S/dt )*(p-p_n)*q*dx)
	F += inner(sym(grad(v)),teff_NH_UJ1(u,lambda_m,mu))*dx - beta * p * nabla_div(v)*dx
	# 
	# Non linear problem definition
	dX0     = TrialFunction(MS)
	J       = derivative(F, X0, dX0)
	Problem = NonlinearProblem(F, X0, bcs = bcs, J = J)
	# 
	#
	###################################################
	######################## Solver ###################
	###################################################
	# 
	# set up the non-linear solver
	solver  = nls.petsc.NewtonSolver(mesh.comm, Problem)
	# Absolute tolerance
	solver.atol = 5e-10
	# relative tolerance
	solver.rtol = 1e-11
	solver.convergence_criterion = "incremental"
	solver.max_it = 10
	# 
	# 
	###################################################
	################ Processing #######################
	###################################################
	# 
	xdmf = XDMFFile(mesh.comm, "case_B.xdmf", "w")
	xdmf.write_mesh(mesh)
	# 
	t = 0
	list_t=[]
	list_react = []
	# list_t.append(t)
	imposed_disp = 0
	print('Begin solve')
	for n in range(num_steps-1):
		dt.value=time_list[n+1]-time_list[n]
		t += dt.value
		list_t.append(t)
		# update BCs
		imp_disp.value = theorical_u[n+1]
		imposed_disp=imp_disp.value
		if MPI.COMM_WORLD.rank == 0:
			print(f"Time step {n+1}/{num_steps}, dt: {dt.value}, Imposed displacement {imposed_disp} m, time = {time_list[n+1]}")
		try:
			num_its, converged = solver.solve(X0)
		except:
			if MPI.COMM_WORLD.rank == 0:
				print("*************") 
				print("Solver failed")
				print("*************") 
			Jcost=1e6
			if MPI.COMM_WORLD.rank == 0:		
				print("Jcost (unit) = ",float(Jcost))
			return float(Jcost)
		X0.x.scatter_forward()
		# Update Value
		Xn.x.array[:] = X0.x.array[:]
		Xn.x.scatter_forward()
		__u, __p = Xn.split()
		# 
		react_local = assemble_scalar(RF_expr)
		react = (mesh.comm.allreduce(react_local, op=MPI.SUM))
		list_react.append(react)
		# 
		# 
		# Export the results
		__u.name = "Displacement"
		__p.name = "Pressure"
		xdmf.write_function(__u,t)
		xdmf.write_function(__p,t)
		# 
		# 
		if ((t >= 18.8) and (t<=75)) or ((t>=97.8) and (t<=145)):
			Jcost+=(RF_exp[n+1]-react)**2
		# 
		# Flux evaluation
		flux_b = -permeability/mu*grad(X0.sub(1))
		# Equivalent to VectorFunctionSpace
		fluxx = FunctionSpace(mesh, VectorElement("CG", mesh.ufl_cell(), 1))
		flux_exp = Expression(flux_b, fluxx.element.interpolation_points())
		q__b     = Function(fluxx)
		q__b.interpolate(flux_exp)
		q__b.name = "Q"
		xdmf.write_function(q__b,t)
		(N0, N1, N2) = (Constant(mesh,(ScalarType(1),ScalarType(0),ScalarType(0))),Constant(mesh,(ScalarType(0),ScalarType(1),ScalarType(0))),
			Constant(mesh,(ScalarType(0),ScalarType(0),ScalarType(1))))
		flux_magnitude = sqrt(dot(flux_b,N0)**2+dot(flux_b,N1)**2+dot(flux_b,N2)**2) 
		fluxx2 = FunctionSpace(mesh, FiniteElement("CG", mesh.ufl_cell(), 1))
		flux_exp2 = Expression(flux_magnitude, fluxx2.element.interpolation_points())
		q__b2     = Function(fluxx)
		q__b2.interpolate(flux_exp)
		q__b2.name = "Q_mag"
		xdmf.write_function(q__b2,t)
	# 
	xdmf.close()	
	# 
	print('End solve')
	#
	if MPI.COMM_WORLD.rank == 0:
		import csv

		def export_to_csv(data, filename):
			with open(filename, 'w', newline='') as file:
				writer = csv.writer(file)
				writer.writerows(data)
			print(f"Data exported to {filename} successfully")

		# Export data to CSV
		export_to_csv(np.transpose([list_t,list_react]), 'RF_B_8mm.csv')
		###################################################
		################ Plots ############################
		###################################################
		# 
		import matplotlib.pyplot as plt
		# 
		plt.rcParams.update({'font.size': 15})
		plt.rcParams.update({'legend.loc':'upper right'})
		# 
		fig1, ax1 = plt.subplots()
		ax1.plot(column1_list,column4_list,linestyle='-',linewidth=2,color='powderblue')
		ax1.set_xlabel('time (s)')
		ax1.set_ylabel('Reaction Force (N)')
		fig1.tight_layout()
		fig1.savefig('Experimental_B.jpg')
		# 
		fig2, ax2 = plt.subplots()
		ax2.plot(list_t,list_react,linestyle='-',linewidth=2,color='powderblue')
		ax2.set_ylabel('Reaction Force (N)')
		ax2.set_xlabel('time (s)')
		fig2.tight_layout()
		fig2.savefig('Model_B.jpg')
		# Superimposed
		fig2, ax2 = plt.subplots()
		ax2.plot(column1_list,column4_list,linestyle='-',linewidth=2,color='lightgreen',label='Exp')
		ax2.plot(list_t,list_react,linestyle=':',linewidth=2,color='olivedrab',label='theo')
		ax2.set_ylabel('Reaction Force (N)')
		ax2.set_xlabel('time (s)')
		ax2.legend()
		fig2.tight_layout()
		fig2.savefig('Model_B_2_8mm.jpg')
	# 
	###################################################
	################ Post-Processing ##################
	###################################################
	# 
	# 
	# Evaluate final time
	end_t = time.time()
	if MPI.COMM_WORLD.rank == 0:		
		print("100*Jcost (unit) = ",100*(math.sqrt((1/num_steps)*float(Jcost))))
	# 
	return math.sqrt((1/num_steps)*float(Jcost))
	# 
if __name__ == "__main__":
	# 
	import time
	import os
	import glob
	import shutil
	# 
	## Create the output directories and files
	# Output directory
	directory = "Results"
	# Parent Directory path
	parent_dir = "./"
	# Path
	path       = os.path.join(parent_dir, directory)
	try:
		os.mkdir(path)
	except:
		pass
	# 
	parameter_file = "./Results/optimized parameters.txt"
	initial_p_file = "./Results/initial_parameters.txt"
	others_file    = "./Results/readme.txt"
	# 
	# 
	# 
	## parameter
	# solution:  3.948701375735829 0.5121016901557495 0.044234830705966045 36.127131162818344
	x0=[3.948701375735829, 0.5121016901557495, 0.044234830705966045, 36.127131162818344]
	eval_opti(x0)