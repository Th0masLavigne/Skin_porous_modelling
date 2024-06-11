# 
# 
def Cauchy(mesh,u,lambda_m,mu):
	"""
	Compute the Cauchy stress from a compressible neo-Hookean formulation: W = (mu / 2) * (Ic - tr(Identity(len(u)))) - mu * ln(J) + (lambda_m / 2) * (ln(J))^2
	Inputs:
	- lambda_m, mu : Lame_Coefficients
	- u : displacement 
	Outputs: 
	- Cauchy stress tensor: Nanson's Formula with diff(W, F), First Piola Kirchoff stress tensor
	"""
	from ufl import variable, Identity, grad, det, tr, ln, diff
	## Deformation gradient
	F = variable(Identity(mesh.geometry.dim) + grad(u))
	J  = variable(det(F))
	## Right Cauchy-Green tensor
	C = variable(F.T * F)
	##Invariants of deformation tensors
	Ic = variable(tr(C))
	## Strain energy density function
	W = (mu / 2) * (Ic - tr(Identity(mesh.geometry.dim))) - mu * ln(J) + (lambda_m / 2) * (ln(J))**2
	return (1/J)*diff(W, F)*F.T
# 
# 
def Cauchy_F(mesh,u,lambda_m,mu,FF):
	"""
	Compute the Cauchy stress from a compressible neo-Hookean formulation: W = (mu / 2) * (Ic - tr(Identity(len(u)))) - mu * ln(J) + (lambda_m / 2) * (ln(J))^2
	Inputs:
	- lambda_m, mu : Lame_Coefficients
	- u : displacement 
	Outputs: 
	- Cauchy stress tensor: Nanson's Formula with diff(W, F), First Piola Kirchoff stress tensor
	"""
	from ufl import variable, Identity, grad, det, tr, ln, diff
	## Deformation gradient
	F = variable(FF)
	# F = variable(Identity(mesh.geometry.dim) + grad(u))
	J  = variable(det(F))
	## Right Cauchy-Green tensor
	C = variable(F.T * F)
	##Invariants of deformation tensors
	Ic = variable(tr(C))
	## Strain energy density function
	W = (mu / 2) * (Ic - tr(Identity(mesh.geometry.dim))) - mu * ln(J) + (lambda_m / 2) * (ln(J))**2
	return (1/J)*diff(W, F)*F.T
# 
# 
def Cauchy_F2(mesh,Xn,X0,lambda_m,mu):
	"""
	Compute the Cauchy stress from a compressible neo-Hookean formulation: W = (mu / 2) * (Ic - tr(Identity(len(u)))) - mu * ln(J) + (lambda_m / 2) * (ln(J))^2
	Inputs:
	- lambda_m, mu : Lame_Coefficients
	- u : displacement 
	Outputs: 
	- Cauchy stress tensor: Nanson's Formula with diff(W, F), First Piola Kirchoff stress tensor
	"""
	from ufl import variable, Identity, grad, det, tr, ln, diff
	## Deformation gradient
	u = X0+Xn
	du = X0
	F = variable(Identity(mesh.geometry.dim) + grad(du))
	J  = variable(det(F))
	## Right Cauchy-Green tensor
	C = variable(F.T * F)
	##Invariants of deformation tensors
	Ic = variable(tr(C))
	## Strain energy density function
	W = (mu / 2) * (Ic - tr(Identity(mesh.geometry.dim))) - mu * ln(J) + (lambda_m / 2) * (ln(J))**2
	return (1/J)*diff(W, F)*F.T
# 
# 
def PK1(mesh,u,lambda_m,mu):
	"""
	Compute the Cauchy stress from a compressible neo-Hookean formulation: W = (mu / 2) * (Ic - tr(Identity(len(u)))) - mu * ln(J) + (lambda_m / 2) * (ln(J))^2
	Inputs:
	- lambda_m, mu : Lame_Coefficients
	- u : displacement 
	Outputs: 
	- Cauchy stress tensor: Nanson's Formula with diff(W, F), First Piola Kirchoff stress tensor
	"""
	from ufl import variable, Identity, grad, det, tr, ln, diff
	## Deformation gradient
	F = variable(Identity(mesh.geometry.dim) + grad(u))
	J  = variable(det(F))
	## Right Cauchy-Green tensor
	C = variable(F.T * F)
	##Invariants of deformation tensors
	Ic = variable(tr(C))
	## Strain energy density function
	W = (mu / 2) * (Ic - tr(Identity(mesh.geometry.dim))) - mu * ln(J) + (lambda_m / 2) * (ln(J))**2
	return diff(W, F)
# 
# 
def read_excel_sheet(file_path, sheet_name):
	"""
	Given the structure of the experimental Data, 
	returns the 4 firtst columns of the sheet sheet_name
	in an xlsx file 
	"""
	import pandas as pd
	# Read the specified sheet from the Excel file
	df = pd.read_excel(file_path, sheet_name=sheet_name)
	# Extract the first four columns as separate lists
	column1 = df.iloc[:, 0].tolist()
	column2 = df.iloc[:, 1].tolist()
	column3 = df.iloc[:, 2].tolist()
	# Return the four lists
	return column1, column2, column3
# 
# 
def read_excel_sheet2(file_path, sheet_name):
	"""
	Given the structure of the experimental Data, 
	returns the 4 firtst columns of the sheet sheet_name
	in an xlsx file 
	"""
	import pandas as pd
	# Read the specified sheet from the Excel file
	df = pd.read_excel(file_path, sheet_name=sheet_name)
	# Extract the first four columns as separate lists
	column1 = df.iloc[:, 0].tolist()
	column2 = df.iloc[:, 1].tolist()
	column3 = df.iloc[:, 2].tolist()
	column4 = df.iloc[:, 3].tolist()
	column5 = df.iloc[:, 4].tolist()
	column6 = df.iloc[:, 5].tolist()
	column7 = df.iloc[:, 6].tolist()
	column8 = df.iloc[:, 7].tolist()
	# Return the four lists
	return column1, column2, column3, column4, column5, column6, column7, column8
# 
# 
def keep_every_n_elements(lst,n):
	"""
	Returns a list composed of the every 
	n^th element of lst.
	"""
	result = []
	for i in range(len(lst)):
		if i % n == 0:
			result.append(lst[i])
	return result
# 
# 
def calculer_moyenne_g(nombre_elements,data):
	"""
	compute the left mean of a part of a list
	"""
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
# 
# 
def calculer_moyenne_d(nombre_elements,data):
	"""
	compute the right mean of a part of a list
	"""
	moyenne_droite = []
	for i in range(len(data)):
		droite = data[i + 1:i + nombre_elements + 1]
		if len(droite)==0:
			moyenne_droite.append(data[i])
		else:
			moyenne_droite.append(calculer_moyenne(droite))
	return moyenne_droite
# 
# 
def calculer_moyenne_c(nombre_elements,data):
	"""
	compute the centered mean of a part of a list
	"""
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
# 
def calculer_moyenne(liste):
	"""
	compute the mean of a list
	"""
	import numpy as np
	if len(liste) == 0:
		return 0
	return np.mean(liste)
# 
# 
def spin_tensor(u):
    """
    Compute spin tensor caused by du
    TENSOR NOTATIONS CAUSE ITS SKEW
    """
    from ufl import skew, grad
    grad_X_du = grad(u)
    w = skew(grad_X_du)  # w is in the reference UPDATED configuration
    return(w)
# 
# 
def export_to_csv(data, filename):
	import csv
	with open(filename, 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerows(data)
	print(f"Data exported to {filename} successfully")
# 
# 
def Elastic(mesh,u,lambda_m,mu):
	"""
	Compute the cauchy stress tensor from the displacement and lame coefficients
	Inputs:  
	- lambda_m, mu : Lame_Coefficients
	- u : displacement
	Outputs: 
	- lambda_m * nabla_div(u) * Identity(len(u)) + 2*mu*sym(grad(u)) : stress tensor 
	"""
	from ufl import sym, grad, nabla_div, Identity
	## Deformation
	epsilon = sym(grad(u))
	return lambda_m * nabla_div(u) * Identity(3) + 2*mu*epsilon