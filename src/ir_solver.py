import numpy as np
from scipy import sparse
import scipy.sparse.linalg as sparse_algebra
from grid import grid
from pprint import pprint
from time import time
from tqdm import tqdm

import sys
from os.path import dirname
sys.path.append(dirname(__file__))
from current_mapgen import current_mapgen
import matplotlib.pyplot as plt

# M1 M4 M7 NandGate
#grid_params= {
#'size_x' : 600000,
#'size_y' : 600000,
#'pitches' : [2560,24000,48000],
#'widths' : [72,192,512],
#'dirs' : [0,1,0],
#'via_res' : [3,3,0],
#'rhos' : [0.2,0.08,0.065],
#'offsets' : [0,4000,4000],
#'unit_micron': 2000
#}

#
grid_params= {
'size_x' : 1300000,
'size_y' : 1300000,
'pitches' : [4800, 56000, 40000, 40000, 40000],
'widths' : [340, 960, 2800, 2800, 2800],
'min_widths' : [340, 960, 2800, 2800, 2800],
'dirs' : [0,1,0,1,0],
'via_res' : [3,3,1,1,0],
'res_per_l' : [0.2,0.08,0.08,0.08,0.065],
'offsets' : [0,4000,4000,4000,4000],
'unit_micron': 2000
}

ir_solver_params = {
'vdd' : 1.1,
'ir_drop_limit' : 0.07
}



fh_current = "./current_map_file.txt"
#TODO read and covert to report or 2d map elsewhere 
class ir_solver:
  def __init__(self, ir_solver_params, grid_params):
    self.grid_params = grid_params
    self.grid = grid(grid_params)
    self.vdd = ir_solver_params['vdd']
    self.ir_drop_limit = ir_solver_params['ir_drop_limit']
  
  @property 
  def internal_grid(self):
    return self.grid
  
  @property 
  def J(self):
    return self._J

  # to return a copy of the grid allowing modifcations
  #def get_grid(self):
    
  def set_grid(self, grid):
    self.grid = grid
    self._G = grid.Gmat

  def generate_Vsrc_nodes(self,vsrc_nodes):
    num_vsrc = len(vsrc_nodes)
    Gmat = self.grid.Gmat 
    num_nodes = self.grid.num_nodes
    size = num_nodes + num_vsrc
    Gmat.resize((size,size))
    top_layer = self.grid.top_layer
    for vsrc_num,vsrc_node in enumerate(vsrc_nodes):
      x = vsrc_node[0]
      y = vsrc_node[1]
      node_h = self.grid.get_node(top_layer, x, y, True)  
      Gmat[node_h.G_loc, num_nodes+vsrc_num] = 1
      Gmat[num_nodes+vsrc_num, node_h.G_loc] = 1

  def create_J(self, current_map, vsrc_nodes, blockages=None):
    blockages_present = False
    if blockages is not None:
      for layers, (x_range,y_range) in blockages:
        if self.grid.bottom_layer in layers:
            blockages_present = True
            
    self._J_nodes= []
    num_vsrc = len(vsrc_nodes)
    num_nodes = self.grid.num_nodes
    size = num_nodes + num_vsrc
    bottom_layer = self.grid.bottom_layer
    J = np.zeros((size,1))
    for current_node in current_map:
      x = current_node[0]
      y = current_node[1]
      I = current_node[2]
      blockage_area = False
      if blockages_present:
        count = 0
        for _, (x_range,y_range) in blockages:
            if (x_range[0] < x < x_range[1]) and (y_range[0] < y < y_range[1]):
              blockage_area = True
              #print(" blocked_area %d"%count)
              #print((x_range,y_range))
              #print(x,y)
            count+=1
      node_h = self.grid.get_node(bottom_layer, x, y, True) 
      if blockage_area :
        node_h.add_current(0)
        J[node_h.G_loc] += 0
      else:
        node_h.add_current(-I)
        J[node_h.G_loc] += -I
      
      self._J_nodes.append(node_h)
        
    for vsrc_num,vsrc_node in enumerate(vsrc_nodes):
      V = vsrc_node[2]
      J[num_nodes+vsrc_num] = V
    
    return sparse.dok_matrix(J)

  def update_J(self, current_map, vsrc_nodes, blockages=None):
    for node_h in self._J_nodes:
      node_h.set_current(0)
    self._J = self.create_J(current_map, vsrc_nodes, blockages)

  def set_J(self,J):
    self._J = J

  def build(self, current_map, vsrc_nodes, grid = None, blockages= None):
    if grid == None:
      self.grid.build(blockages)
      #print(self.grid.num_nodes)
      self.generate_Vsrc_nodes(vsrc_nodes)
    elif grid != None:
      self.grid = grid
      print("ERROR: unknown condition")
    self._G = self.grid.Gmat 
    self._J = self.create_J(current_map,vsrc_nodes, blockages)

  def solve_ir(self,
                ir_map_file = None,
                J_map_file = None,
                grid= None,
                regions=None):
    #print("starting to solve")
    st  = time();
    if grid is None:
      G = sparse.dok_matrix.tocsc(self._G)
    else:
      G = sparse.dok_matrix.tocsc(grid.Gmat)  
    #print("load G time %f"%(time()-st))
    st  = time();
    I = sparse.identity(self._G.shape[0]) * 1e-13
    G = G + I
    #print("update G time %f"%(time()-st))
    st  = time();
    J = sparse.dok_matrix.tocsc(self._J)
    #print("load J time %f"%(time()-st))
    st  = time();
    V = sparse_algebra.spsolve(G, J, permc_spec='COLAMD', use_umfpack=False)# , permc_spec=None, use_umfpack=True)
    #print("solve V time %f"%(time()-st))
    st  = time();
    nodes = self.grid.nodes
    solution = np.zeros((0,3))
    current = np.zeros((0,3))
    unit_micron = self.grid_params['unit_micron']
    worst_case_ir = 0
    if regions is not None:
        region_ir = np.zeros(len(regions))
    else:
        region_ir = None
    #print(regions)
    #print("initial ",region_ir)
    for node_num,node_h in enumerate(nodes):
      if(node_h.has_stripe): 
        node_h.set_voltage(V[node_num]) 
        if(node_h.layer == self.grid.bottom_layer):
          ir_drop =  self.vdd - V[node_num]
          solution = np.append(solution,
                             [[node_h.x/unit_micron, 
                               node_h.y/unit_micron, 
                               self.vdd - V[node_num]]],
                             axis=0)
          current = np.append(current,[[node_h.x/unit_micron, 
                               node_h.y/unit_micron, 
                               -node_h.current]],
                             axis=0)
          if regions is not None:
            for n, region in enumerate(regions):
                x_range = region[0]
                y_range = region[1]
                if(    (x_range[0] < node_h.x < x_range[1]) 
                   and (y_range[0] < node_h.y < y_range[1])):
                   region_ir[n] = max(region_ir[n], ir_drop)
                   #print(n,region_ir[n],x_range,y_range,node_h.x,node_h.y)
                   break
    worst_case_ir = np.max(solution[:,2])

    #print("regional IR") 
    #print(region_ir)
    #print("store solution/current time %f"%(time()-st))
    st  = time();
    if ir_map_file != None:
      np.savetxt(ir_map_file, solution, fmt='%8.3f, %8.3f, %8.6e') 
    if J_map_file != None:
      np.savetxt(J_map_file, current, fmt='%8.3f, %8.3f, %8.6e') 
    #for val in solution:
    #  print(val)
    #print("Worst Case IR %f\n"%worst_case_ir)
    return solution, worst_case_ir, region_ir
 

vsrc_nodes_temp = np.array([
                            [150*2000,150*2000,1.1],
                            [150*2000,300*2000,1.1],
                            [150*2000,450*2000,1.1],
                            [300*2000,150*2000,1.1],
                            [300*2000,300*2000,1.1],
                            [300*2000,450*2000,1.1],
                            [450*2000,150*2000,1.1],
                            [450*2000,300*2000,1.1],
                            [450*2000,450*2000,1.1],
                            [150*2000, 50*2000,1.1],
                            [550*2000, 50*2000,1.1]
                            ])
macro_layers = [0,1,2]

#macro1 = (macro_layers, ((400000,500000),(640000,840000)))
#macro2 = (macro_layers, ((600000,700000),(640000,840000)))
#macro3 = (macro_layers, ((800000,900000),(640000,840000)))
#blockages = [ macro1, macro2, macro3]

macro1 = (macro_layers, ((125*2000,175*2000),(400*2000,600*2000)))
macro2 = (macro_layers, (( 25*2000,150*2000),(350*2000,400*2000)))
macro3 = (macro_layers, (( 25*2000,150*2000),(290*2000,340*2000)))
macro4 = (macro_layers, (( 25*2000,150*2000),(230*2000,280*2000)))
blockages = [ macro1, macro2, macro3, macro4]

#blockages = [([0, 1, 2], ((396000, 478000), (308000, 588000))),
#             ([0, 1, 2], ((1058000,1212000 ), (930000, 978000))),
#             ([0, 1, 2], ((310000,528000), ( 566000, 654000)))]
random_map_params = {
'x_dim' : 600,
'y_dim' : 600,
'var' : 1,
'max_cur' : 1e-6,
'len_scale' : 100,
'num_maps' : 1
}
current_map_params = {
'unit_micron' : 2000
}

if __name__ == '__main__':
  t0 = time()
  #TODO current_map_processing VSRC processing
  
  #curr_mapgen_h = current_mapgen(current_map_params)
  #maps = curr_mapgen_h.gen_random_maps(random_map_params)
  #maps_dok = curr_mapgen_h.to_dok_map(maps)
  t = time()
  print("end of map gen %f"%(t-t0))
  
  st = time()
  print(" ir start_time %f"%(st-t0))
  ir_solver_h = ir_solver(ir_solver_params,grid_params)
  
  #map_dok = np.loadtxt('aes_data.rpt',delimiter=',')
  map_dok = np.loadtxt('Sample_cur.csv',delimiter=',')
  map_dok[:,0:2] = map_dok[:,0:2]*2000
  map_dok[:,0:2] = map_dok[:,0:2].astype(int)

  #map_dok[:,2] = 1e-6

  #np.savetxt('curr_map.txt',map_dok,delimiter=',')
  
  #ir_solver_h.build(map_dok,vsrc_nodes_temp)  
  ir_solver_h.build(map_dok,vsrc_nodes_temp, blockages =blockages )  
  #ir_solver_h.build(maps_dok[0],vsrc_nodes_temp)  

  t = time()
  print("build time %f"%(t - st))

  new_grid = ir_solver_h.internal_grid
  template_map = {}
  template_map['region1'] = { 'x_range' : ( 600000,1300000),
                              'y_range' : (      0, 600000),
                              'layers' : [ (1, 56000*8)] 
                              }
  template_map['region2'] = { 'x_range' : (      0, 600000),
                              'y_range' : ( 600000,1300000), 
                              'layers' :  [ (1, 56000*8)],
                              }
  template_map['region3'] = { 'x_range' : ( 600000,1300000),
                              'y_range' : ( 600000,1300000),
                              'layers' :  [ (1, 56000*8)],
                              }
  regions = [] 
  for _,region in tqdm(template_map.items()):
    x_range = region['x_range']
    y_range = region['y_range']
    regions.append((x_range,y_range))
 #   for layer_info in region['layers']:
 #     layer, pitch = layer_info
 #     new_grid.update_layer_stripes(layer, x_range, y_range, pitch)
  
  ir_map_file = 'ir_map_dok.csv'
  J_map_file = 'J_map_dok.csv'

  print("create irregular grid time : %d"%(time() - t))
  t= time()
  print(regions)
  V,_,region_ir = ir_solver_h.solve_ir(ir_map_file,J_map_file,grid=new_grid,
                regions = regions)
  print(region_ir)
  print("run time %f"%(time() - t))
  #fig,ax = plt.subplots()
  #im = ax.imshow(fields)
  #fig.colorbar(im)
  #plt.show(block = False)
  plt.show()
  ir_map_file = 'no_macro_ir_map_dok.csv'
  J_map_file = 'no_macro_J_map_dok.csv'
  ir_solver_h = ir_solver(ir_solver_params,grid_params)
  ir_solver_h.build(map_dok,vsrc_nodes_temp) 
  ir_solver_h.update_J(map_dok,vsrc_nodes_temp,blockages) 
  V,_,region_ir = ir_solver_h.solve_ir(ir_map_file,J_map_file)

  
  
