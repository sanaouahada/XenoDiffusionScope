import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import scipy.interpolate
from matplotlib.patches import Circle, Rectangle
from tqdm import tqdm

# Plots should be handled in separate Plotter file.

class TopArray:
    def __init__(self, tpc, mesh,model,path_to_patterns,path_to_model):
        self.model = model
        self.tpc = tpc
        self.path_to_patterns = path_to_patterns
        self.mesh = mesh
        self.path_to_model = path_to_model
        
        self.make_results_grid()
        
        
    def make_results_grid(self, x_step=1, y_step=1):
        self.grid_x_step = x_step
        self.grid_y_step = y_step
        self.grid_x_max = self.tpc.radius
        self.grid_y_max = self.tpc.radius
        self.grid_x_min = -self.tpc.radius
        self.grid_y_min = -self.tpc.radius
        self.grid_x = np.arange(self.grid_x_min, 
                                self.grid_x_max,
                                self.grid_x_step)
        self.grid_y = np.arange(self.grid_y_min,
                                self.grid_y_max,
                                self.grid_y_step)
        self.grid_xx,self.grid_yy = np.meshgrid(self.grid_x,
                                                self.grid_y,
                                                indexing='ij')
        self.grid_zz = np.zeros((self.grid_yy.shape))
        return None
    
    def load_top_array(self):
        with open('%s/%s.pck'%(self.path_to_model, self.model), 'rb') as file:
            self.sensors = pickle.load(file)
        self.n_sensors = len(self.sensors)
        
        return None
    
    def plot_toparray(self,ax, pe_in_sensors = False):
        ax.set_aspect('equal')
        for _quad_i,_quad in enumerate(self.sensors):
            xy = (_quad[0],_quad[2])
            width = _quad[1]-_quad[0]
            height = _quad[3]-_quad[2]
            if pe_in_sensors == False:
                ax.add_patch(Rectangle(xy,width,height, 
                                       fill = False, 
                                       color = 'k'))
            else:
                #TO DO
                #check if results exist and default to False
                
                cm = plt.get_cmap('gnuplot')
                log_pe = np.log10(self.pe_in_sensors)
                results_max = np.max(log_pe)
                results_min = np.min(log_pe[np.isfinite(log_pe)])
                log_pe = np.nan_to_num(log_pe, copy = True,neginf=results_min)
                results_max = np.max(log_pe)
                results_min = np.min(log_pe)
                pe = log_pe[_quad_i]

                ax.add_patch(Rectangle(xy,width,height, 
                                       fill = True, 
                                       edgecolor = 'k',
                                       facecolor = cm((pe-results_min)/
                                                      (results_max-results_min)),
                                      )
                            )
        ax.set_ylim(-90,90)
        ax.set_xlim(-90,90)
        ax.set_aspect('equal')
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.add_patch(Circle((0,0),75, color = 'r',fill = False))
        return ax
    
    def integrate_in_quad(self,quad):
        ans = self.results_interp.integral(quad[0],quad[1],quad[2],quad[3])
        return ans
    
    def load_pattern(self, hex_id):
        with open(self.path_to_patterns + 'hex_v0_%d.pck'%hex_id, 'rb') as file:
            pattern = pickle.load(file)
        return pattern
    
    def load_all_patterns(self, all_at_once = False):
        if all_at_once:
            with open(self.path_to_patterns+'all_paterns.pck', 'rb') as file:
                patterns = pickle.load(file)
            self.patterns = patterns
            return None
        
        patterns = dict()
        for _hex_i in tqdm(range(self.mesh.n_hexes),
                           'Loading LCE patterns.',
                           total = self.mesh.n_hexes):
            patterns[_hex_i] = self.load_pattern(_hex_i)
        
        self.patterns = patterns
        
        return None
    
    def fill_grid_from_events(self, counts_pe_on_hex):
        '''
        Sum over all the patterns of LCE x number of pe per hex to
        get the final pe pattern in the array.
        '''
        
        for hex_i in tqdm(range(self.mesh.n_hexes),
                          'Summing all normalized patterns. 1+1+2+3+5+8+...',
                          total = self.mesh.n_hexes):
            #pattern = load_pattern(self, hex_i)
            pattern = self.patterns[hex_i]
            if counts_pe_on_hex[hex_i] > 0:
                _zz = pattern.ev(self.grid_xx,self.grid_yy) * counts_pe_on_hex[hex_i]
                self.grid_zz += _zz
            else:
                continue
        
        self.results_interp = scipy.interpolate.RectBivariateSpline(
            self.grid_x, 
            self.grid_y,
            self.grid_zz,
            s = 0)
        return None
    
    def n_pe_in_sensors(self):
        pe_in_sensors = np.apply_along_axis(self.integrate_in_quad,
                                          1, 
                                          self.sensors)
        self.pe_in_sensors = pe_in_sensors
        return np.abs(pe_in_sensors)
        