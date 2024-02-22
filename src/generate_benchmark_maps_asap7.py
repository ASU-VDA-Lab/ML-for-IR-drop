from ir_solver import ir_solver
import numpy as np
from time import time
from tqdm.auto import tqdm
import math
import matplotlib
import matplotlib.pyplot as plt
import sys
import scipy
from glob import glob
import cv2
import skimage.transform
from current_mapgen import current_mapgen
from scipy.interpolate import griddata
import os

def load_current_maps(technology, tech_area_res, designs,location):
  global current_scale
  current_maps = []
  print("Loading reference designs")
  pbar = tqdm(designs,leave=False)
  design_sum, design_size= 0, 0
  design_areas =[]
  for n, design in enumerate(pbar):
    fname = '%s/%s/%s/current_map.csv'%(location, technology, design)
    current_map = np.loadtxt(fname, delimiter=',') * current_scale[n]
    scale = int(1/tech_area_res)
    x,y = current_map.shape
    current_map = np.sum(current_map.reshape(x//scale,scale,y//scale,scale),axis=(1,3))
    #current_map = scipy.ndimage.gaussian_filter(current_map, 2.5)
    pbar.set_postfix({'Last shape': current_map.shape})
    design_sum += np.sum(current_map)
    design_size += current_map.size
    design_areas.append(current_map.shape)
    #plt.figure()
    #plt.imshow(current_map,cmap='jet')
    #plt.colorbar() 
    current_maps.append(current_map)
  mean = design_sum/design_size
  return current_maps, mean, design_areas

def load_gan_maps(technology, areas, mean,location):
  global rng
  print("Loading GAN designs")
  designs = glob('%s/%s/current_map*.png'%(location,technology))
  #print(designs)

  gan_maps = []
  gan_designs = []
  pbar = tqdm(designs,leave=False)
  for design in pbar:
    fname = os.path.splitext(os.path.basename(design))[0]
    gan_im = cv2.imread(design, cv2.COLORMAP_JET)
    #pbar.set_postfix({'Last shape': gan_im.shape})
    gan_maps.append(gan_im)
    gan_designs.append(fname)
  gan_maps = np.array(gan_maps)
  gan_maps_scaled = gan_maps*mean/np.mean(gan_maps)
  
  current_maps = []
  for gan_im in tqdm(gan_maps_scaled,leave=False):
    area_num = rng.integers(len(areas))
    resized_gan_im = skimage.transform.resize(gan_im, areas[area_num])
    current_maps.append(resized_gan_im)
  
  print(len(current_maps))
  #for num in tqdm(rng.integers(len(current_maps),size=5)):
  #  plt.figure()
  #  plt.imshow(current_maps[num],cmap='jet')
  #  plt.colorbar() 
  return current_maps, gan_designs

def load_params():
  ir_solver_params = {
  'vdd' : 0.7,
  'ir_drop_limit' : 0.007
  }
  unit_micron = 2000
  layers_name = ['M2', 'M5', 'M6', 'M7', 'M8']
  grid_params= {
    'unit_micron': unit_micron,
    'size_x' : 2000, # temporary updated later
    'size_y' : 2000, # temporary updated later
    'pitches' : [1080, 6000, 12000, 4096, 6400],
    'dirs' : [0,1,0,1,0],
    #'via_res' : [3,3,1,1,1,0],
    'res_per_l' : [1.83, 1.83, 1.83, 1.83, 0.065],
    'widths' : [36, 240, 576, 1024, 1600],
    'min_widths' : [36, 48, 64, 64,80],
    'via_res' : [10, 10, 10, 10, 10],
    'offsets' : [60,60,60, 60, 60]
    }

  templates = [] 
  for l1 in [1,2,3,4]:
      templates.append([(1,l1*6000)])
  default_template = 3
  region_size = 100
  return (ir_solver_params, unit_micron, grid_params, templates,
          default_template, region_size, layers_name )

def main():
  global rng
  global current_scale
#################################################################################################
  #location_designs = "design_current_map_20210428"
  location_designs = "designs"
  location_gan = "1k_generated_images"
  location_results = "results"
  technology = "asap7"
  tech_area_res = 0.2
  #designs = [ "aes", "bp", "bp_be", "bp_fe", "bp_multi", "dynamic_node",
  #            "ibex", "jpeg", "swerv", "swerv_wrapper" ]
  #designs = [ "aes", "ibex", "jpeg", "dynamic_node" ]
  current_scale = [ 0.1, 1, 1, 0.05 ]
  designs = [ "dynamic_node"]
  #designs = [ "aes"]
  
####################################################################################################


  seed = "".join(designs)
  seed = "".join(["%d"%ord(c) for c in seed])
  seed = int(seed)
  print("Designs: %s,\n seed:\n %d"%(",".join(designs),seed))
  rng = np.random.default_rng(seed)

  print("Loading design files")
  design_current_maps, design_mean, design_areas = load_current_maps(technology,
                                                                     tech_area_res, 
                                                                     designs,
                                                                     location_designs)
  print(design_mean)
 #############################################################################################
 #############################################################################################
 #############################################################################################
  #TODO  
  # Commented out so as not to rerun, but uncomment if running for the
  # first time. 
  generate_outputs(design_current_maps, designs,location_results, technology, False, )
 #############################################################################################
 #############################################################################################
 #############################################################################################
 #############################################################################################
  print("Loading GAN generated files")
  # uncomment when making gan
  #gan_current_maps, gan_designs= load_gan_maps(technology, design_areas, design_mean, location_gan)
  #generate_outputs(gan_current_maps[0:1000], gan_designs[0:1000],location_results,
  #                 technology, True )

def generate_outputs(current_maps, designs, location_results, technology, generated=False):
  pbar = tqdm(current_maps) 
  if generated:
    folder = 'generated'
  else:
    folder = 'designs'
  for n, current_map in enumerate(pbar):
    (ir_solver_params, unit_micron, grid_params, templates, default_template,
     region_size, layers_name) = load_params()
    x_dim = current_map.shape[0]
    y_dim = current_map.shape[1]
    print(f"Dimensions: {x_dim}, {y_dim}")
    grid_params['size_x'] = x_dim*unit_micron
    grid_params['size_y'] = y_dim*unit_micron
    curr_mapgen_h = current_mapgen(grid_params)
    
    pbar.set_postfix({'Status': '%15s'%'Generate Dok'})
    res_fname = location_results  + '/%s/%s/%s_current_map'%(technology,
                                                                 folder,
                                                                 designs[n])
    np.savetxt(res_fname+'.csv', current_map, fmt='%5.3e', delimiter=',')
    fig = plt.figure()
    plt.imshow(current_map.T,cmap='jet')
    plt.xlabel('width')
    plt.ylabel('height')
    plt.colorbar()
    fname = res_fname+'.png'
    plt.title(fname)
    plt.savefig(fname)

    current_map_dok = curr_mapgen_h.to_dok_map(current_map)
    pbar.set_postfix({'Status': '%15s'%'Generate vsrcs'})
    vsrc_nodes = generate_vsrc( ir_solver_params['vdd'],
                                x_dim,
                                y_dim,
                                unit_micron)
#    print('num_vsrc',len(vsrc_nodes))
    pbar.set_postfix({'Status': '%15s'%'Generate regions'})
    regions, reg_state, irreg_state = generate_regions(
                                      len(templates), 
                                      region_size,
                                      default_template,
                                      (x_dim,y_dim),
                                      unit_micron)

    pbar.set_postfix({'Status': '%15s'%'Generate regular IR'})
    res_fname = location_results  + '/%s/%s/%s_voltage_map_regular'%(technology,
                                                                     folder,
                                                                     designs[n])
    vol, reg_grid = solve_ir(ir_solver_params,
             grid_params,
             vsrc_nodes,
             current_map_dok[0],
             regions,
             reg_state,
             templates,
             res_fname
             )
    pbar.set_postfix({'Status': '%15s'%'Generate regular spice'})
    res_fname = location_results  + '/%s/%s/%s_reg_grid.sp'%(technology,folder,designs[n])
    generate_spice(reg_grid.internal_grid, reg_grid.J, 'n1',
                   ir_solver_params['vdd'], res_fname, layers_name)

    pbar.set_postfix({'Status': '%15s'%'Generate regular IR'})
    res_fname = location_results  + '/%s/%s/%s_voltage_map_irregular'%(technology,
                                                                       folder,
                                                                       designs[n])
    #vol, irreg_grid = solve_ir(ir_solver_params,
    #         grid_params,
    #         vsrc_nodes,
    #         current_map_dok[0],
    #         regions,
    #         irreg_state,
    #         templates,
    #         res_fname
    #         )

    #pbar.set_postfix({'Status': '%15s'%'Generate irregular spice'})
    #res_fname = location_results  + '/%s/%s/%s_irreg_grid.sp'%(technology,
    #                                                           folder,
    #                                                           designs[n])
    #generate_spice(irreg_grid.internal_grid, irreg_grid.J, 'n1',
    #              ir_solver_params['vdd'], res_fname, layers_name)




def generate_spice(grid, currents, net_name, voltage, file_name, layers_name):
#     fig_num = plt.gcf().number
  gmat = grid.Gmat
  gmat_nodes = grid.nodes
  res_count = 0
  vol_count = 0
  cur_count = 0
  with open(file_name,'w') as f:
    for n, row in enumerate(tqdm(gmat.items(),leave=False)):
      loc, val = row
      node1_loc, node2_loc = loc
      if node1_loc >= node2_loc: #ignore lower triangle
        continue
      if node1_loc >= grid.num_nodes or node2_loc >= grid.num_nodes: #vias
        if node1_loc < grid.num_nodes:
            via_node = gmat_nodes[node1_loc]
        else:
            via_node = gmat_nodes[node2_loc]
        n1 = '%s_%s_%d_%d'%(net_name, layers_name[via_node.layer], via_node.x, via_node.y)
        n2 = '0'
        f.write('V%d %s %s %f \n'%(vol_count, n1, n2, voltage))
        vol_count +=1
      else:
        node1 = gmat_nodes[node1_loc]
        node2 = gmat_nodes[node2_loc]
        n1 = '%s_%s_%d_%d'%(net_name, layers_name[node1.layer], node1.x, node1.y)
        n2 = '%s_%s_%d_%d'%(net_name, layers_name[node2.layer], node2.x, node2.y)
        f.write('R%d %s %s %f \n'%(res_count, n1, n2, 1/-val))
        res_count += 1
    for n,(loc,current) in enumerate(tqdm(currents.items(),leave=False)):
      if loc[0]<grid.num_nodes and abs(current)>1e-15:
        current_node = gmat_nodes[loc[0]]
        n1 = '%s_%s_%d_%d'%(net_name, layers_name[current_node.layer], current_node.x, current_node.y)
        n2 = '0'
        f.write('I%d %s %s %e \n'%(cur_count, n1, n2, -current))
        cur_count +=1
    f.write('.op\n.end')

def solve_ir(ir_solver_params,
             grid_params,
             vsrc_nodes,
             current_map,
             regions,
             grid_state,
             templates,
             res_fname
             ):
  grid_h = ir_solver(ir_solver_params,grid_params)
  grid_h.build(current_map,vsrc_nodes) 
  grid = grid_h.internal_grid
  #regular grid template 3
  for n,template in enumerate(grid_state):
    x_range = regions[n][0]
    y_range = regions[n][1]
    layers_info = templates[template]
    grid.update_region( x_range, y_range, layers_info)
  solution, max_drop, region_ir = grid_h.solve_ir(
                                    regions=regions
                                    )
  print(f"Dimensions V1: {np.max(solution[:,0])}, {np.max(solution[:,1])}")
  print(f"Dimensions V2: {np.min(solution[:,0])}, {np.min(solution[:,1])}")
  voltages = save_heat_map(solution, res_fname, grid_params, False)
  plot_hist(solution[:,2], res_fname, num_bins=20,plot=False)
  return voltages, grid_h

def save_heat_map(vol_dok, res_fname, grid_params, plot= False):
  ## target grid to interpolate to
  print(f"Dimensions VOL1: {np.max(vol_dok[:,0])}, {np.max(vol_dok[:,1])}")
  print(f"Dimensions VOL2: {np.min(vol_dok[:,0])}, {np.min(vol_dok[:,1])}")
  x_dim = grid_params['size_x']/grid_params['unit_micron']
  y_dim = grid_params['size_y']/grid_params['unit_micron']
  xd = np.arange(0,x_dim,1)
  yd = np.arange(0,y_dim,1)
  print(xd,yd)
  yi,xi = np.meshgrid(yd,xd)
  print(xi,yi)
  ## interpolate
  zi_vol = griddata((vol_dok[:,0],vol_dok[:,1]),vol_dok[:,2],(xi,yi),
                     method='cubic')
  print(zi_vol)
  nans = np.isnan(zi_vol)
  if nans.any():
    # for each nan point, find its nearest neighbor
    inds = np.argmin(
        ((xi.flatten())[:, None] - xi[nans])**2 +
        ((yi.flatten())[:, None] - yi[nans])**2 +
        (nans.flatten())[:,None]*100000
        ,axis=0)
    # ... and use its value
    inds = np.unravel_index(inds, nans.shape)
    zi_vol[nans] = zi_vol[inds]
  print(zi_vol)
  fname = res_fname+'.csv'
  np.savetxt(fname, zi_vol, fmt='%5.3e', delimiter=',')
  fig = plt.figure()
  plt.imshow(zi_vol.T,cmap='jet')
  plt.xlabel('width')
  plt.ylabel('height')
  plt.title(fname)
  plt.colorbar()
  fname = res_fname+'.png'
  plt.savefig(fname)
  if plot == False:
    plt.close(fig)
  return zi_vol

def generate_regions(num_templates, 
                     region_size,
                     default_template,
                     shape,
                     unit_micron):
  regions = []
  x_dim, y_dim = shape
  NUM_REGIONS_X = math.ceil(x_dim/region_size)
  NUM_REGIONS_Y = math.ceil(y_dim/region_size)
  NUM_REGIONS = NUM_REGIONS_X * NUM_REGIONS_Y
  
  init_state = rng.integers(0, num_templates, size=NUM_REGIONS)
  for x in range(NUM_REGIONS_X):
    for y in range(NUM_REGIONS_Y):
      n = x*NUM_REGIONS_Y + y
      lx, ux = x*region_size, (x+1)*region_size
      ly, uy = y*region_size, (y+1)*region_size
      cx, cy = (lx+ux)/2, (ly+uy)/2
      lx,ux,ly,uy = lx*unit_micron,ux*unit_micron,ly*unit_micron,uy*unit_micron
      regions.append(((lx,ux),(ly,uy)))
  reg_grid_state = default_template*np.ones(NUM_REGIONS,dtype='int')
  #print('regular_grid',reg_grid_state)
  #print('irregular_grid',init_state)
  return regions, reg_grid_state, init_state

def generate_vsrc(vdd,x_dim,y_dim, unit_micron):
  vsrc_pitch  = 30 
  vsrc_offset_x = 15
  vsrc_offset_y = 15
  
  num_bump_x = int((x_dim - vsrc_offset_x) /vsrc_pitch)+1
  num_bump_y = int((y_dim - vsrc_offset_y) /vsrc_pitch)+1
  num_bumps = num_bump_x*num_bump_y
  
  
  bump_mat = np.array([[x+y*num_bump_x for x in range(num_bump_x)]for y in range(num_bump_y)])
  bump_list = []
  for x in range(0,num_bump_x,2):
    for y in range(0,num_bump_y,2):
        bump_list.append(bump_mat[y:y+2,x:x+2].reshape(-1))
        
  bump_list = np.array(bump_list)
  vsrc_node_locs = []
  for row in bump_list:
    vsrc_node_locs.append(rng.choice(row))
    
  vsrc_nodes = []
  for vsrc_node_loc in vsrc_node_locs:
    lx = int(vsrc_node_loc/num_bump_y) 
    ly = vsrc_node_loc % num_bump_y
    llx = (vsrc_offset_x + lx*vsrc_pitch)*unit_micron
    lly = (vsrc_offset_y + ly*vsrc_pitch)*unit_micron
    vsrc_nodes.append([llx,lly,vdd])
  return vsrc_nodes

def plot_hist(vals,res_fname,num_bins=20,plot =False):
  fig = plt.figure()
  # An "interface" to matplotlib.axes.Axes.hist() method
  n, bins, patches = plt.hist(x=vals, bins=num_bins, color='#0504aa',
                              alpha=0.7)#, rwidth=0.85)
  plt.grid(axis='y', alpha=0.75)
  plt.xlabel('Voltages')
  plt.ylabel('Frequency')
  plt.title('IR Drop Histogram')
  xloc = 0.75*np.max(vals)
  yloc = 0.75*np.max(n)
  plt.text(xloc, yloc, r'$\mu=$%5.3e'%(np.mean(vals)))
  yloc = 0.65*np.max(n)
  plt.text(xloc, yloc, r'$max=$%5.3e'%(np.max(vals)))
  #print('%5.3e %5.3e'%(np.mean(vals),np.max(vals)))
  maxfreq = n.max()
  # Set a clean upper y-axis limit.
  plt.ylim(ymax=np.ceil(maxfreq / 1000) * 1000 if maxfreq % 1000 else maxfreq + 1000)
  fname = res_fname+'_hist.png'
  plt.savefig(fname) 
  if plot == False:
    plt.close(fig)

  cdf = np.cumsum(n)
  cdf = np.insert(cdf,0,0)
  fig =plt.figure()
  plt.plot(bins,cdf)
  fname = res_fname+'_cdf.png'
  plt.savefig(fname) 
  if plot == False:
    plt.close(fig)
  return bins,cdf

if __name__ == "__main__":
  main()
  plt.show()
