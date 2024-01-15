import gmsh
from mpi4py       import MPI
import numpy as np
import sys
import os
# 
# Dircetories for save
directory_geo = "Geo_file"
# Parent Directory path
parent_dir = "./"
# Path
path       = os.path.join(parent_dir, directory_geo)
try:
	os.mkdir(path)
except:
	pass
directory_mesh = "Mesh_file"
# Parent Directory path
parent_dir = "./"
# Path
path       = os.path.join(parent_dir, directory_mesh)
try:
	os.mkdir(path)
except:
	pass
# 
###################################################
################## User defined ###################
###################################################
# 
# Specify the name for the exported mesh
# 
mesh_filename = "Symmetric_skin_XY_RF"
# 
# Geometrical parameters [m] (Remove real 0 in case of division by the coordinates)
# 
zx,zy,zz = 1e-16, 1e-16, 1e-16
Top_layer_thickness    = 2e-3
Bottom_layer_thickness = 8e-3
Patch_dimension        = 8e-3
L0                     = 36e-3
ROI_y                  = 24e-3
ROI_x                  = L0 + 2*17e-3
U_tot_y                = 24e-3
U_tot_x                = 17e-3
u_middle_y             = 10e-3
# u_middle_y             = 8e-3
u_middle_x             = 8e-3
# u_middle_x             = 9e-3
u_side_y               = 7e-3
# u_side_y               = 8e-3
u_side_x               = 17e-3
offset_y               = 3*ROI_y
offset_x               = ROI_x
# 
# open_gmsh_result for visualizing in the GUI : True or False
# 
open_gmsh_result = True
# 
# Mesh parameters
# 
# lc_coarse = (Top_layer_thickness+Bottom_layer_thickness)/3
# lc_inter = 1e-3
# lc_hypoderm = 5e-4
# lc_patch = 2.5e-4
# Coarse for debug
lc_coarse = 1e-2 #(Top_layer_thickness+Bottom_layer_thickness)/3
lc_inter = 5e-3
lc_hypoderm = 2e-3
lc_patch = 1e-3
lc_left = 4e-4
lc_local = 1e-3
# For threshold local refinement
dit_min_ROI_b = Top_layer_thickness
dist_max_ROI_b=2*Top_layer_thickness
dit_min_ROI = 0.6*Patch_dimension
dist_max_ROI=1.5*Patch_dimension
dist_min_patch=2*Patch_dimension
dist_max_patch=4*Patch_dimension
dist_min_left = 3e-3
dist_max_left = 3e-2
dist_min_local_refine = 5e-3
dist_max_local_refine = 1e-2
# 
# 
###################################################
################## Computation  ###################
###################################################
# 
# 
# Code made for parallel running
# 
mesh_comm = MPI.COMM_WORLD
model_rank = 0
# 
gmsh.initialize()
# 
gdim = 3
# 
if mesh_comm.rank == model_rank:
	gmsh.model.mesh.createGeometry()
	# 
	# top_offset        = gmsh.model.occ.addBox(zx, zy, zz+Bottom_layer_thickness, ROI_x/2+offset_x, ROI_y/2+offset_y, Top_layer_thickness)
	top_offset        = gmsh.model.occ.addBox(zx, zy, zz+Bottom_layer_thickness, ROI_x/2+offset_x, np.pi*47e-3/2, Top_layer_thickness)
	# bottom_offset     = gmsh.model.occ.addBox(zx, zy, zz, ROI_x/2+offset_x, ROI_y/2+offset_y, Bottom_layer_thickness)
	bottom_offset     = gmsh.model.occ.addBox(zx, zy, zz, ROI_x/2+offset_x, np.pi*47e-3/2, Bottom_layer_thickness)
	top_ROI           = gmsh.model.occ.addBox(zx, zy, zz+Bottom_layer_thickness, ROI_x/2, ROI_y/2, Top_layer_thickness)
	bottom_ROI        = gmsh.model.occ.addBox(zx, zy, zz, ROI_x/2, ROI_y/2, Bottom_layer_thickness)
	gmsh.model.occ.synchronize()
	# 
	# Remove duplicate entities and synchronize
	# 
	gmsh.model.occ.removeAllDuplicates()
	# 
	right_patch_box       = gmsh.model.occ.addBox(zx+ROI_x/2-U_tot_x, zy, zz, Patch_dimension, Patch_dimension/2, Bottom_layer_thickness+Top_layer_thickness)
	# 
	gmsh.model.occ.synchronize()
	# 
	# Remove duplicate entities and synchronize
	# 
	gmsh.model.occ.removeAllDuplicates()
	# 
	right_u_middle    = gmsh.model.occ.addBox(zx+ROI_x/2-u_middle_x, zy, zz+Bottom_layer_thickness, u_middle_x, u_middle_y/2, Top_layer_thickness)
	right_u_top       = gmsh.model.occ.addBox(zx+ROI_x/2-u_side_x, zy+ROI_y/2-u_side_y, zz+Bottom_layer_thickness, u_side_x, u_side_y, Top_layer_thickness)
	# 
	gmsh.model.occ.synchronize()
	# Remove duplicate entities and synchronize
	gmsh.model.occ.removeAllDuplicates()
	gmsh.model.occ.synchronize()
	# 
	# Save Geometry for debug
	gmsh.write(directory_geo+'/'+mesh_filename+'_Geom.geo_unrolled')
	# 
	surfaces, volumes = [gmsh.model.getEntities(d) for d in [ gdim-1, gdim]]
	# Refinement regions & Facet tags
	local_refine = []
	right_patch_marker, right_marker, left_marker, front_marker, back_marker, bottom_marker, top_marker, u_right_marker, right_side_patch_marker = 1,2,3,4,5,6,7,8,9
	right_patch, ROI, left, right, front, back, bottom, top, u_right, ROI_bottom, right_side_patch = [], [], [], [], [], [], [], [], [], [], []
	for boundary in surfaces:
		center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
		if np.isclose(center_of_mass[0], zx):
		 	left.append(boundary[1])
		elif np.isclose(center_of_mass[0], zx+ROI_x/2+offset_x):
		 	right.append(boundary[1])
		elif np.isclose(center_of_mass[2], zz):
		 	bottom.append(boundary[1])
		elif np.isclose(center_of_mass[2], zz+Bottom_layer_thickness+Top_layer_thickness) and ((center_of_mass[0]>zx+ROI_x/2) or (center_of_mass[0]<zx+ROI_x/4)): 
			top.append(boundary[1])
		elif np.isclose(center_of_mass[1], zy+ROI_y/2+offset_y):
		 	front.append(boundary[1])
		# back 
		elif np.isclose(center_of_mass[1], zy):
			back.append(boundary[1])
		elif np.isclose(center_of_mass[0], zx+ROI_x/2-U_tot_x+Patch_dimension/2) and np.isclose(center_of_mass[2], zz+Bottom_layer_thickness+Top_layer_thickness):
		 	right_patch.append(boundary[1])
		elif ((np.isclose(center_of_mass[0],zx+ROI_x/2-u_side_x/2) and np.isclose(center_of_mass[2],zz+Bottom_layer_thickness+Top_layer_thickness)) 
			or (np.isclose(center_of_mass[0],zx+ROI_x/2-u_middle_x/2) and np.isclose(center_of_mass[2],zz+Bottom_layer_thickness+Top_layer_thickness))) :
			u_right.append(boundary[1])
		elif ((np.isclose(center_of_mass[1], zy+ROI_y/2) and np.isclose(center_of_mass[2],zz+Bottom_layer_thickness+Top_layer_thickness/2)) or
			(np.isclose(center_of_mass, [zx+ROI_x/4, zy+ROI_y/2, zz+Bottom_layer_thickness/2]).all()) or
			(np.isclose(center_of_mass[0], zx+ROI_x/2) and np.isclose(center_of_mass[2],zz+Bottom_layer_thickness/2)) or 
			(np.isclose(center_of_mass[0], zx+ROI_x/2) and np.isclose(center_of_mass[2],zz+Bottom_layer_thickness+Top_layer_thickness/2))):
			ROI.append(boundary[1]) 
		# Add bottom roi 
		if np.isclose(center_of_mass[2], zz+Bottom_layer_thickness) and (center_of_mass[1] < ROI_y/2):
			ROI_bottom.append(boundary[1])
		# identify the left side and top of patch and u for constraint concentration
		if (np.isclose(center_of_mass[0], zx+ROI_x/2-U_tot_x) and np.isclose(center_of_mass[2],zz+Bottom_layer_thickness+Top_layer_thickness/2)):
		 	local_refine.append(boundary[1])
		elif  (np.isclose(center_of_mass[1], zy+ROI_y/2) and np.isclose(center_of_mass[2],zz+Bottom_layer_thickness+Top_layer_thickness/2) and center_of_mass[0]>zx+ROI_x/2-U_tot_x):
			local_refine.append(boundary[1])
		if np.isclose(center_of_mass[0], zx+ROI_x/2-U_tot_x+Patch_dimension) and np.isclose(center_of_mass[1], zy+Patch_dimension/4):
			right_side_patch.append(boundary[1])
	# 
	# DEBUG
	# 
	try:
		boundaries = gmsh.model.getBoundary(volumes, oriented=False)
		assert len(boundaries)==(len(right_patch)+len(left)+len(right)+len(front)+len(back)+len(bottom)+len(top)+len(u_right))
		print("All Boundaries have been properly identified")
	except:
		all_b = [elem[-1] for elem in boundaries]
		identified_b = right_patch+left+right+front+back+bottom+top+u_right
		forgot = [ x for x in all_b if x not in identified_b ]
		print(f"Forgotten boundaries : {forgot}")
	#
	# 
	# Cell_tags
	top_layer, bottom_layer = [], []
	top_layer_marker, bottom_layer_marker = 100, 200
	for element in volumes:
		center_of_mass = gmsh.model.occ.getCenterOfMass(element[0], element[1])
		if np.isclose(center_of_mass[2], Bottom_layer_thickness+Top_layer_thickness/2):
			top_layer.append(element[1])
		elif np.isclose(center_of_mass[2], Bottom_layer_thickness/2):
			bottom_layer.append(element[1])
	# 
	# Define the tags (Physical entities)
	# Facets
	tdim =2
	gmsh.model.addPhysicalGroup(tdim, u_right, u_right_marker)
	gmsh.model.setPhysicalName(tdim, u_right_marker, 'u_right')
	# 
	gmsh.model.addPhysicalGroup(tdim, right_patch, right_patch_marker)
	gmsh.model.setPhysicalName(tdim, right_patch_marker, 'right_patch')
	# 
	gmsh.model.addPhysicalGroup(tdim, left, left_marker)
	gmsh.model.setPhysicalName(tdim, left_marker, 'left')
	# 
	gmsh.model.addPhysicalGroup(tdim, right, right_marker)
	gmsh.model.setPhysicalName(tdim, right_marker, 'right')
	# 
	gmsh.model.addPhysicalGroup(tdim, bottom, bottom_marker)
	gmsh.model.setPhysicalName(tdim, bottom_marker, 'bottom')
	# 
	gmsh.model.addPhysicalGroup(tdim, top, top_marker)
	gmsh.model.setPhysicalName(tdim, top_marker, 'top_face')
	# 
	gmsh.model.addPhysicalGroup(tdim, front, front_marker)
	gmsh.model.setPhysicalName(tdim, front_marker, 'front')
	# 
	gmsh.model.addPhysicalGroup(tdim, back, back_marker)
	gmsh.model.setPhysicalName(tdim, back_marker, 'back')
	gmsh.model.addPhysicalGroup(tdim, right_side_patch, right_side_patch_marker)
	gmsh.model.setPhysicalName(tdim, right_side_patch_marker, 'Right_side_patch')
	# 
	# Cells
	gmsh.model.addPhysicalGroup(gdim, bottom_layer, bottom_layer_marker)
	gmsh.model.setPhysicalName(gdim, bottom_layer_marker, 'bottom_layer')
	#
	gmsh.model.addPhysicalGroup(gdim, top_layer, top_layer_marker)
	gmsh.model.setPhysicalName(gdim, top_layer_marker, 'top_layer') 
	# 
	# 
	#  Meshing: set distance fields to faces for local refinement
	# 
	# Patch
	distance = gmsh.model.mesh.field.add("Distance")
	gmsh.model.mesh.field.setNumbers(distance, "FacesList", right_patch)
	# The next step is to use a threshold function vary the resolution from these surfaces in the following way:
	threshold = gmsh.model.mesh.field.add("Threshold")
	gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
	gmsh.model.mesh.field.setNumber(threshold, "LcMin", lc_patch)
	gmsh.model.mesh.field.setNumber(threshold, "LcMax", lc_coarse)
	gmsh.model.mesh.field.setNumber(threshold, "DistMin", dist_min_patch)
	gmsh.model.mesh.field.setNumber(threshold, "DistMax", dist_max_patch)
	# 
	# ROI
	# 
	roi_distance = gmsh.model.mesh.field.add("Distance")
	gmsh.model.mesh.field.setNumbers(roi_distance, "FacesList", ROI)
	# The next step is to use a threshold function vary the resolution from these surfaces in the following way:
	roi_l = gmsh.model.mesh.field.add("Threshold")
	gmsh.model.mesh.field.setNumber(roi_l, "IField", roi_distance)
	gmsh.model.mesh.field.setNumber(roi_l, "LcMin", lc_inter)
	gmsh.model.mesh.field.setNumber(roi_l, "LcMax", lc_coarse)
	gmsh.model.mesh.field.setNumber(roi_l, "DistMin", dit_min_ROI)
	gmsh.model.mesh.field.setNumber(roi_l, "DistMax", dist_max_ROI)
	# 
	# ROI_bottom
	# 
	roi_b_distance = gmsh.model.mesh.field.add("Distance")
	gmsh.model.mesh.field.setNumbers(roi_b_distance, "FacesList", ROI_bottom)
	# The next step is to use a threshold function vary the resolution from these surfaces in the following way:
	roi_b = gmsh.model.mesh.field.add("Threshold")
	gmsh.model.mesh.field.setNumber(roi_b, "IField", roi_b_distance)
	gmsh.model.mesh.field.setNumber(roi_b, "LcMin", lc_hypoderm)
	gmsh.model.mesh.field.setNumber(roi_b, "LcMax", lc_coarse)
	gmsh.model.mesh.field.setNumber(roi_b, "DistMin", dit_min_ROI_b)
	gmsh.model.mesh.field.setNumber(roi_b, "DistMax", dist_max_ROI_b)
	# 
	# Constraint concentration
	# roi_lc_distance = gmsh.model.mesh.field.add("Distance")
	# gmsh.model.mesh.field.setNumbers(roi_lc_distance, "FacesList", local_refine)
	# # The next step is to use a threshold function vary the resolution from these surfaces in the following way:
	# roi_lc = gmsh.model.mesh.field.add("Threshold")
	# gmsh.model.mesh.field.setNumber(roi_lc, "IField", roi_lc_distance)
	# gmsh.model.mesh.field.setNumber(roi_lc, "LcMin", lc_local)
	# gmsh.model.mesh.field.setNumber(roi_lc, "LcMax", lc_coarse)
	# gmsh.model.mesh.field.setNumber(roi_lc, "DistMin", dist_min_local_refine)
	# gmsh.model.mesh.field.setNumber(roi_lc, "DistMax", dist_max_local_refine)
	# 
	# Keep the smallest mesh_size:
	minimum = gmsh.model.mesh.field.add("Min")
	gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", [threshold,roi_l,roi_b]) # add other fields in the list if needed
	gmsh.model.mesh.field.setAsBackgroundMesh(minimum)
	# 
	# Set kernel attributes
	# beware min is 1e-3 here by default, you need to change it
	print(gmsh.option.getNumber("Mesh.MeshSizeMin"))
	print(gmsh.option.getNumber("Mesh.MeshSizeMax"))
	gmsh.model.occ.synchronize()
	gmsh.option.setNumber("General.Terminal",1)
	gmsh.option.setNumber("Mesh.Optimize", True)
	gmsh.option.setNumber("Mesh.OptimizeNetgen", True)
	gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
	gmsh.model.occ.synchronize()
	# gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
	# gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
	gmsh.option.setNumber("Mesh.MeshSizeMin", min(lc_patch,lc_inter,lc_local))
	gmsh.option.setNumber("Mesh.MeshSizeMax", lc_coarse)
	# gmsh.option.setNumber("Mesh.MshFileVersion", 2.0)
	# 
	# Mesh
	gmsh.model.mesh.generate(gdim)
	gmsh.write(directory_mesh+'/'+mesh_filename+".msh")
	if open_gmsh_result:
		if 'close' not in sys.argv:
			gmsh.fltk.run()
	# 
	print("Min mesh size allowed ", gmsh.option.getNumber("Mesh.MeshSizeMin"))
	print("Max mesh size allowed ", gmsh.option.getNumber("Mesh.MeshSizeMax"))
	gmsh.finalize()
	# 
	# 
	from dolfinx.io import gmshio, XDMFFile
	import dolfinx
	mesh, cell_tags, facet_tags = gmshio.read_from_msh(directory_mesh+'/'+mesh_filename+".msh", MPI.COMM_WORLD, 0, gdim=3)
	# maximum element size (for comparison with the equivalent code of FEniCS)
	tdim      = mesh.topology.dim
	num_cells = mesh.topology.index_map(tdim).size_local
	# h         = dolfinx.cpp.mesh.h(mesh, tdim, range(num_cells))
	# dh        = h.max()
	# dh2       = h.min()
	# if MPI.COMM_WORLD.rank == 0:
	# 	print('dh_max=',dh, 'min',dh2)
	# FOR DEBUG SAVE FIRST XDMF
	with XDMFFile(MPI.COMM_WORLD, directory_mesh+'/'+mesh_filename+"_tag.xdmf", "w") as xdmftag:
			xdmftag.write_mesh(mesh)
			xdmftag.write_meshtags(facet_tags, mesh.geometry)
			xdmftag.write_meshtags(cell_tags, mesh.geometry)
	# 
exit()