import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import vtk
from vtk.util.numpy_support import vtk_to_numpy

#####################
#### setup latex ####
#####################

pgf_with_latex = {                      # setup matplotlib to use latex for output
    'pgf.texsystem': 'pdflatex',        # change this if using xetex or lautex
    'text.usetex': True,                # use LaTeX to write all text
    'font.size' : 10,
    #'font.family': 'sans-serif',
    'font.family': 'serif',
    #'font.serif': ['Computer Modern Roman'],  # blank entries should cause plots
    'font.serif': [],                          # blank entries should cause plots 
    'font.sans-serif': [],                     # to inherit fonts from the document
    'font.monospace': [],
    'text.latex.preamble':r'\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage[detect-all]{siunitx}\usepackage{amsmath}\usepackage{bm}'
    #'text.latex.preamble':r'\usepackage{sansmath}\sansmath\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage[detect-all]{siunitx}\usepackage{amsmath}\usepackage{bm}'
    }

mpl.rcParams.update(pgf_with_latex)

plt.rcParams["axes.axisbelow"] = False ### draw axes, ticks and labels always above everything elese

##########################
#### define functions ####
##########################

def load_vtk_plane(FileName,a):
    reader = vtk.vtkGenericDataObjectReader()
    reader.SetFileName(FileName)

    reader.Update()

    vtk_array = reader.GetOutput().GetPointData().GetArray(a) #0 for vel., 1 for vort.
    vtk_numpy_array = vtk_to_numpy(vtk_array)
    return vtk_numpy_array

def convert_velocity_field(data_vel, nx, ny, Lx, Ly):
    x = np.linspace(0,Lx,nx)
    y = np.linspace(0,Ly,ny)
    
    #print('Shape of x:\t', x.shape)
    #print('Shape of y:\t', y.shape)

    u = data_vel[:,0].reshape(y.shape[0], x.shape[0])
    #print('Shape of u:\t', u.shape)

    v = data_vel[:,1].reshape(y.shape[0], x.shape[0])
    #print('Shape of v:\t', v.shape)

    speed = np.sqrt(u*u + v*v)
    #print('Shape of speed: ', speed.shape, '\n\n')
    
    return x, y, u, v, speed

###################################
#### define literature results ####
###################################


ghia_re100_u = np.array([[1.,1.],[.9766,.84123],[.9688,.78871],[.9609,.73722],[.9531,.68717],[.8516,.23151],[.7344,.00332],[.6172,-.13641],[.5,-.20581],[.4531,-.21090],[.2813,-.15662],[.1719,-.1015],[.1016,-.06434],[.0703,-.04775],[.0625,-.04192],[.0547,-.03717],[0.,0.]])
ghia_re100_v = np.array([[1.,0.],[.9688,-.05906],[.9609,-.07391],[.9531,-.08864],[.9453,-.10313],[.9063,-.16914],[.8594,-.22445],[.8047,-.24533],[.5,.05454],[.2344,.17527],[.2266,.17507],[.1563,.16077],[.0938,.12317],[.0781,.1089],[.0703,.10091],[.0625,.09233],[0,0]])

###################
#### load data ####
###################

### parameters

R_s = 1
V_s = 1
D = 1

dx = 1/126.*R_s

Nx = 126
Ny = 126
Lx = Nx*dx
Ly = Ny*dx


print('=='*20)
################################

directory = './output/'

################################

file_lbm_re100 = directory+'cavity_Vel_00010.vtk'

data_vel_lbm_re100 = np.delete(load_vtk_plane(file_lbm_re100,0),2,1)
print('data_vel_lbm_re100:\t',data_vel_lbm_re100.shape)

x_lbm_re100, y_lbm_re100, u_lbm_re100, v_lbm_re100, speed_lbm_re100 = convert_velocity_field(data_vel_lbm_re100,Nx,Ny,Lx,Ly)

################################


##############
#### plot ####
##############

### colors
cmap = plt.get_cmap('Blues')
color_ghia = 'black'
color_line = 'black'
color_nast_x = 'tab:cyan'
color_nast_y = 'crimson'
color_lbm_x = 'tab:olive'
color_lbm_y = 'royalblue'

### parameters
every = 4

lw = 1.75
arrowsize = 1.
den = 1.25

im_interpolation = 'nearest'

legend_fontsize = 6

markersize=3
markeredgewidth=1


###########################################################
fig = plt.figure(figsize=(7.22433,2.75))
gs = gridspec.GridSpec(nrows=2, ncols=7, wspace=0., hspace=1.75, width_ratios=(.1,.35,1.,.5,1.,.2,.1))
###########################################################
ax00 = fig.add_subplot(gs[:2,0])
ax00.axis('off')
ytext = .325
ax00.text(0.,ytext,r'$\text{Re} = 100$', rotation=90.)
###########################################################
ax02 = fig.add_subplot(gs[:,6])

normV = mpl.colors.Normalize(vmin=0., vmax=1.)
cbarV = fig.colorbar(mpl.cm.ScalarMappable(norm=normV, cmap=cmap), cax=ax02, ticks=[0.,.25,.5,.75,1.], orientation='vertical', label='Fluid Velocity $V$')
cbarV.ax.tick_params(axis='y', direction='in')
ax02.set_yticklabels([r'\num{0}',r'',r'',r'',r'\num{1}'])
###########################################################
ax1 = fig.add_subplot(gs[:2,2])
ax1.tick_params(direction='in', which='both', bottom=True, top=True, left=True, right=True)
ax1.set_title(r'LBM')
ax1.set_aspect('equal')

strm1 = ax1.streamplot(x_lbm_re100,y_lbm_re100,u_lbm_re100,v_lbm_re100,color=speed_lbm_re100,arrowsize=arrowsize,linewidth=lw,cmap=cmap,density=den)

ax1.hlines(.5,0,1, color=color_line, alpha=.2)
ax1.vlines(.5,0,1, color=color_line, alpha=.2)

### ticks x
tick_loc_maj = np.round(np.linspace(0.,1.,3),decimals=1)
tick_loc_min = np.round(np.linspace(0.,1.,5),decimals=1)

ax1.set_xticks(tick_loc_maj, minor=False)
ax1.set_xticks(tick_loc_min, minor=True)
ax1.set_xticklabels(tick_loc_maj)

ax1.set_yticks(tick_loc_maj, minor=False)
ax1.set_yticks(tick_loc_min, minor=True)
ax1.set_yticklabels(tick_loc_maj)

ax1.set_xbound(lower=0., upper=Lx)
ax1.set_ybound(lower=0., upper=Ly)
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$y$')

#ax1.legend()
###########################################################
ax2 = fig.add_subplot(gs[:2,4])
ax2.tick_params(direction='in', which='both', bottom=True, top=True, left=True, right=True)
ax2.minorticks_on()

ax2.plot(ghia_re100_u[:,0],ghia_re100_u[:,1],color=color_ghia,marker='o',markersize=markersize, markeredgewidth=markeredgewidth,fillstyle='full',linestyle='none',clip_on=False,label=r'Ghia 1982')
ax2.plot(ghia_re100_v[:,0],ghia_re100_v[:,1],color=color_ghia,marker='o',markersize=markersize, markeredgewidth=markeredgewidth,fillstyle='full',linestyle='none',clip_on=False)

ax2.plot(x_lbm_re100,u_lbm_re100[:,int(u_lbm_re100.shape[0]/2.)],'--',color=color_lbm_x,label=r'LBM $u(0.5,y)$')
ax2.plot(y_lbm_re100,v_lbm_re100[int(v_lbm_re100.shape[0]/2.),:],'--',color=color_lbm_y,label=r'LBM $v(x,0.5)$')

ax2.set_xbound(lower=0., upper=Lx)
ax2.set_ybound(lower=-.45, upper=1.)
ax2.set_xlabel(r'Coordinate')
ax2.set_ylabel(r'Velocity component')

ax2.legend(fontsize=legend_fontsize)
###########################################################
plt.subplots_adjust(left=.01, right=.95, top=.925, bottom=.15)
fig.savefig('./test.pdf', transparent=True, dpi=600)
