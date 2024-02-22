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
  current_maps = []
  print("Loading reference designs")
  pbar = tqdm(designs,leave=False)
  design_sum, design_size= 0, 0
  design_areas =[]
  for design in pbar:
    fname = '%s/%s/%s/current_map.csv'%(location, technology, design)
    current_map = np.loadtxt(fname, delimiter=',')
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
  print("Loading GAN designs")
  
  gan_maps = []
  gan_designs = []
  seeds = {}
  gan_inputs = np.arange(100)
  #gan_inputs = [62,96]#[12,18,62,96]
  for n in tqdm(gan_inputs,leave=False):
    design = f'{location}/{technology}/current_map{n:02d}.png'
    gan_im = cv2.imread(design, cv2.COLORMAP_JET)
    fname = os.path.splitext(os.path.basename(design))[0]
    gan_maps.append(gan_im)
    gan_designs.append(fname)
    seeds[fname] = n
  gan_maps = np.array(gan_maps)
  gan_maps_scaled = gan_maps*mean/np.mean(gan_maps)
  
  current_maps = []
  areas = np.array(areas)
  min_area, max_area = areas.min(), areas.max()
  for n, gan_im in enumerate(tqdm(gan_maps_scaled,leave=False)):
    rng = np.random.default_rng(seeds[gan_designs[n]])
    chip_size = rng.integers(min_area, max_area)
    resized_gan_im = skimage.transform.resize(gan_im,(chip_size, chip_size))
    current_maps.append(resized_gan_im)
  print(len(current_maps))
  return current_maps, gan_designs, seeds

def load_params():
  ir_solver_params = {
  'vdd' : 1.1,
  'ir_drop_limit' : 0.01
  }
  unit_micron = 2000
  layers_name = ['m1', 'm4', 'm7', 'm8', 'm9']
  grid_params= {
    'unit_micron': unit_micron,
    'size_x' : 2000, # temporary updated later
    'size_y' : 2000, # temporary updated later
    'pitches' : [4800, 28000, 80000, 22400, 22400],
    'dirs' : [0,1,0,1,0],
    #'via_res' : [3,3,1,1,1,0],
    'res_per_l' : [5.42,2,0.1857, 0.075, 0.03],
    'widths' : [340, 960, 2800, 5600, 5600],
    'min_widths' : [140, 280, 800, 800,1600],
    'via_res' : [15, 9, 1, 1, 1],
    'offsets' : [0,4000,4000, 4000, 4000]
    }

  templates = [] 
  for l1 in [1,2,3,4]:
      templates.append([(1,l1*28000)])
  default_template = 3
  region_size = 100
  return (ir_solver_params, unit_micron, grid_params, templates,
          default_template, region_size, layers_name )

def main():
#################################################################################################
  location_designs = "designs"
  location_gan = "generated"
  location_results = "results"
  technology = "nangate45"
  tech_area_res = 0.2
  designs = [ "aes",  "bp_be", "bp_fe", "dynamic_node", "bp", "bp_multi",
              "ibex", "jpeg", "swerv", "swerv_wrapper" ] # Only these work. bp and bp_multi dont
    
####################################################################################################
  print("Loading design files")
  design_current_maps, design_mean, design_areas = load_current_maps(technology,
                                                                     tech_area_res, 
                                                                     designs,
                                                                     location_designs)
  print(design_mean)
  gan_current_maps, gan_designs, seeds= load_gan_maps(technology, design_areas, design_mean, location_gan)
  generate_outputs(gan_current_maps, gan_designs,location_results,
                   technology, seeds, True )


def generate_outputs(current_maps, designs, location_results, technology, seeds, generated=False):
  plot = True
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
    grid_params['size_x'] = x_dim*unit_micron
    grid_params['size_y'] = y_dim*unit_micron
    curr_mapgen_h = current_mapgen(grid_params)
    
    pbar.set_postfix({'Status': '%15s'%'Generate Dok'})
    res_fname = location_results  + '/%s/%s/%s_current'%(technology,
                                                             folder,
                                                             designs[n])
    print("current_map characteristics")
    print(current_map.max())
    print(current_map.min())
    save_and_plot(res_fname, current_map, plot)

    current_map_dok = curr_mapgen_h.to_dok_map(current_map)
    pbar.set_postfix({'Status': '%15s'%'Generate vsrcs'})
    seed = seeds[designs[n]]
    vsrc_nodes = generate_vsrc( ir_solver_params['vdd'],
                                x_dim,
                                y_dim,
                                seed,
                                unit_micron)
#    print('num_vsrc',len(vsrc_nodes))
    pbar.set_postfix({'Status': '%15s'%'Generate regions'})
    regions, reg_state, irreg_state = generate_regions(
                                      len(templates), 
                                      region_size,
                                      default_template,
                                      (x_dim,y_dim),
                                      seed,
                                      unit_micron)

    pbar.set_postfix({'Status': '%15s'%'Generate effective distance map'})
    res_fname = location_results  + '/%s/%s/%s_eff_dist'%(technology,folder,designs[n])
    generate_eff_dist_map(res_fname,grid_params,vsrc_nodes)

    pbar.set_postfix({'Status': '%15s'%'Generate region map'})
    res_fname = location_results  + '/%s/%s/%s_regions'%(technology,folder,designs[n])
    generate_region_maps(res_fname, grid_params, regions, irreg_state)
    
    pbar.set_postfix({'Status': '%15s'%'Generate irregular IR'})
    res_fname = location_results  + '/%s/%s/%s_voltage'%(technology,
                                                                       folder,
                                                                       designs[n])
    vol, irreg_grid = solve_ir(ir_solver_params,
             grid_params,
             vsrc_nodes,
             current_map_dok[0],
             regions,
             irreg_state,
             templates,
             res_fname
             )

    pbar.set_postfix({'Status': '%15s'%'Generate irregular spice'})
    res_fname = location_results  + '/%s/%s/%s.sp'%(technology,
                                                               folder,
                                                               designs[n])
    generate_spice(irreg_grid.internal_grid, irreg_grid.J, 'n1',
                  ir_solver_params['vdd'], res_fname, layers_name)




def generate_spice(grid, currents, net_name, voltage, file_name, layers_name):
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
  print(res_fname)
  failing = solution[:,2]>0.5 
  print(np.sum(failing))
  print(solution[failing,:])
  assert np.all(solution[:,2]>=0)
  assert np.all(solution[:,2]<ir_solver_params['vdd'])
  voltages = save_heat_map(solution, res_fname, grid_params, False)
  return voltages, grid_h

def save_heat_map(vol_dok, res_fname, grid_params, plot= False):
  ## target grid to interpolate to
  x_dim = grid_params['size_x']/grid_params['unit_micron']
  y_dim = grid_params['size_y']/grid_params['unit_micron']
  xd = np.arange(0,x_dim,1)
  yd = np.arange(0,y_dim,1)
  yi,xi = np.meshgrid(yd,xd)
  ## interpolate
  zi_vol = griddata((vol_dok[:,0],vol_dok[:,1]),vol_dok[:,2],(xi,yi),
                     method='cubic')
  zi_vol[zi_vol<0] = 0 
  nans = np.isnan(zi_vol)
  if nans.any():
    # for each nan point, find its nearest neighbor
    locs = np.stack((xi[nans], yi[nans]), axis=-1)
    inds = []
    for n,loc in enumerate(locs):
      x,y = (int(k) for k in loc)
      nearest = np.argmin((xi-x)**2 + (yi-y)**2 + nans*100000)
      nearest = np.unravel_index(nearest,nans.shape)
      zi_vol[x,y] = zi_vol[nearest]
  save_and_plot(res_fname, zi_vol, plot)
  return zi_vol

def generate_region_maps(res_fname, grid_params, regions, region_value, plot=False):
  um = grid_params['unit_micron']
  x_dim = int(grid_params['size_x']/um)
  y_dim = int(grid_params['size_y']/um)
  region_map = np.zeros((x_dim,y_dim))
  for n, region in enumerate(regions):
    x_r, y_r = region
    lx, ux, ly, uy = int(x_r[0]/um), int(x_r[1]/um), int(y_r[0]/um), int(y_r[1]/um)  
    region_map[lx:ux+1,ly:uy+1] = region_value[n]

  save_and_plot(res_fname, region_map, plot)

def save_and_plot(res_fname, in_map, plot=False):
  fname = res_fname+'.csv'
  np.savetxt(fname, in_map, fmt='%5.5e', delimiter=',')
  fig = plt.figure()
  plt.imshow(in_map.T,cmap='jet')
  plt.xlabel('width')
  plt.ylabel('height')
  plt.title(fname)
  plt.colorbar()
  fname = res_fname+'.png'
  plt.savefig(fname)
  if plot == False:
    plt.close(fig)

def generate_regions(num_templates, 
                     region_size,
                     default_template,
                     shape,
                     seed,
                     unit_micron):
  regions = []
  x_dim, y_dim = shape
  NUM_REGIONS_X = math.ceil(x_dim/region_size)
  NUM_REGIONS_Y = math.ceil(y_dim/region_size)
  NUM_REGIONS = NUM_REGIONS_X * NUM_REGIONS_Y
  
  rng = np.random.default_rng(seed)
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

def generate_vsrc(vdd,x_dim,y_dim, seed, unit_micron):
  vsrc_pitch  = 50 
  vsrc_offset_x = 25
  vsrc_offset_y = 25
  
  num_bump_x = int((x_dim - vsrc_offset_x) /vsrc_pitch)+1
  num_bump_y = int((y_dim - vsrc_offset_y) /vsrc_pitch)+1
  num_bumps = num_bump_x*num_bump_y
  
  
  bump_mat = np.array([[x+y*num_bump_x for x in range(num_bump_x)]for y in range(num_bump_y)])
  bump_list = []
  for x in range(0,num_bump_x,3):
    for y in range(0,num_bump_y,3):
        bump_list.append(bump_mat[y:y+3,x:x+3].reshape(-1))
        
  bump_list = np.array(bump_list)
  vsrc_node_locs = []
  rng = np.random.default_rng(seed)
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

def generate_eff_dist_map(res_fname,grid_params,vsrc_nodes, plot=False):
  x_dim = int(grid_params['size_x']/grid_params['unit_micron'])
  y_dim = int(grid_params['size_y']/grid_params['unit_micron'])
  eff_dist_map = np.zeros((x_dim,y_dim))
  for x in range(x_dim):
    for y in range(y_dim):
      d_inv = 0
      for vsrc_node in vsrc_nodes:
        x_loc = vsrc_node[0]/grid_params['unit_micron']
        y_loc = vsrc_node[1]/grid_params['unit_micron']
        d = np.sqrt((x-x_loc)**2 + (y-y_loc)**2)
        if d == 0:
          d_inv = -1
          break
        else:
          d_inv = d_inv+ 1/d
      if d_inv <0 :
        eff_dist_map[x,y] = 0.0
      else:
        eff_dist_map[x,y] = 1/d_inv
  save_and_plot(res_fname, eff_dist_map, plot)

if __name__ == "__main__":
  main()
  plt.show()
