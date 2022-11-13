import numpy as np
import warnings
import scipy.integrate
import scipy.interpolate
import pickle

#optional
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from tqdm import tqdm

#non-trivial packages
from hexalattice.hexalattice import create_hex_grid

class TPC:
    '''
    General TPC class. Contains useful constants, common variables 
    and functions relatesd to the overall system.
    '''
    
    def __init__(self, radius, length, liquid_gap, gas_gap, drift_field):
        self.radius = radius #radius of TPC
        self.length = length #lenght of liquid bellow gate
        self.liquid_gap = liquid_gap #distance from gate to liquid-gas interface
        self.gas_gap = gas_gap #distance from liquid-gas interface to sensors
        self.drift_field = drift_field #V/cm
        
        self.z_gate = 0 # should come from config
        self.liquid_level = self.z_gate + self.liquid_gap
        self.z_anode = self.liquid_level + self.gas_gap
        
        self.drift_velocity = self.model_velocity(self.drift_field)
        self.diffusion_long = self.model_diffusion_longitudinal(self.drift_field)
        self.diffusion_trans = self.model_diffusion_transversal(self.drift_field)
        
        self.mesh_pitch = 1.56 # should come from config
        self.gate_mesh = MeshGrid(self.radius, self.mesh_pitch)
    
    @classmethod
    def model_velocity(self, drift_field):
        '''
        Drift velocity model from NEST, implemented by Yanina 
        Biondi (https://github.com/YaniBion).
        Given a drift_field values [V/cm] returns the expected electron drift 
        velocity in LXe. In mm/us
        '''
        #par = [-3.1046, 27.037, -2.1668, 193.27, -4.8024, 646.04, 9.2471]
        par = [-1.5000, 28.510, -.21948, 183.49, -1.4320, 1652.9, 2.884]
        dv = (par[0] * np.exp(-drift_field/par[1]) + 
              par[2] * np.exp(-drift_field/par[3]) + 
              par[4] * np.exp(-drift_field/par[5]) +
              par[6]) #mm/us
        return dv
    
    @classmethod
    def model_diffusion_longitudinal(self, drift_field):
        '''
        Longitudinal diffusion model: TO DO.
        For 100 V/cm -> ~26cm2/us from 1T
        '''
        if drift_field != 100:
            warnings.warn('Only the long. diffusino value for 100 V/cm is implemented, going with it for now.')
        D_l = 26 #cm2/s
        return D_l * 1e-4 #mm2/us
    
    @classmethod
    def model_diffusion_transversal(self, drift_field):
        '''
        Transversal diffusion model. Linear fit interpolation from the EXO200 paper data (Fig. 7) [
        Curve fit: m=0.010635; b=52.888942
        Linregress: m=0.013021; b=52.752949
        '''
        m=0.010635
        b=52.888942
        ans = m * drift_field + b #cm2/s
        return ans * 1e-4 #mm2/us
    
    @staticmethod
    def get_r(x,y):
        '''Get r from x and y.'''
        return np.sqrt(np.power(x,2) + np.power(y,2))
    
    @staticmethod
    def get_theta(x,y):
        '''Get theta from x and y.'''
        theta = np.arctan(y/x)
        if (x < 0) and (y < 0):
            theta = -theta
        return theta
    
    @staticmethod
    def get_xyz(r,theta,phi):
        '''Convert to cartesian following Physics standard spherical 
        coordinates.'''
        x = r*np.cos(phi)*np.sin(theta)
        y = r*np.sin(phi)*np.sin(theta)
        z = r*np.cos(theta)
        return x,y,z
    
class MeshGrid():
    '''
    Grid-related stuff. Construct, change, focus, etc.
    These helpstrings are becoming worse and worse, I know.
    '''
    
    def __init__(self, r_max, hex_side):
        self.r_max = r_max
        self.hex_side = hex_side
        self.hex_r = self.hex_side * np.sqrt(3)
        
        self.hex_centers = self.construct_mesh()
        self.n_hexes = len(self.hex_centers)
        
    def construct_mesh(self):
        '''
        Construct the hexagonal gate mesh middles.
          * r_max: radius of the electrode
          * a: size of the hexagons
        '''
        
        n_hex_x = 2 * np.ceil(75/self.hex_r)
        n_hex_y = 2 * np.ceil(75/(np.sqrt(2)*self.hex_side))
        
        hex_centers, _ = create_hex_grid(nx=n_hex_x, ny=n_hex_y, 
                                         min_diam = self.hex_side,
                                         crop_circ = self.r_max,
                                         do_plot = False)
        
        return hex_centers
    
    def distance_to_reference_grid(self,pos_xy):
        '''
        Given an array of reference positions (hex grid in this partivular case) - format 
        [[xs,ys]] -, return the distance of a point (x0,y0) to these reference points.
        pos_xy is a 1d 2 value array.
        '''
        hex_distance = np.sqrt(np.power(self.hex_centers[:,0]-pos_xy[0],2) + 
                               np.power(self.hex_centers[:,1]-pos_xy[1],2)
                              )
        return hex_distance

    def get_closest_hex(self, pos_xy, value = True):
        '''
        Given a reference grid and a point on the plane, return the grid index where 
        the point is closeste to a grid point.
        '''
        distances = self.distance_to_reference_grid(pos_xy)
        idx_min_dist = np.argmin(distances)
        if value == True:
            return self.hex_centers[idx_min_dist]
        else:
            return idx_min_dist
    
    def focus_on_grid(self,pos_array):
        '''
        Focuses the electrons to the center of the hexagons. `pos_array` must
        be of the shape (N, 2), where pos_array[:,0] are x values and 
        pos_array[:,1] are y values.
        Returns a (N,2) array of the positions after the focus effect.
        '''
        pos_focus = np.apply_along_axis(self.get_closest_hex, 1, pos_array)
        return pos_focus
    
class XeLamp:
    '''
    Properties and functions of the Xe lamp and its brightly 
    shining effects on producing electrons on the photocathode.
    '''
    
    def __init__(self,delta_t_lamp):
        self.numerical_aperture = 0.22
        self.distance2photok = 2 # mm
        self.length_xy = self.numerical_aperture*self.distance
        
        #time step to consider when reconstructing the lamp pulse
        #use 2 ns (0.002 us) to match ADC freq or 0.25 for testing 
        #(from YB's original code)
        self.delta_t_lamp = delta_t_lamp 
        self.times_lamp = np.linspace(0,6,6/self.delta_t_lamp)
        
    @classmethod
    def pulse_lamp(cls,t):
        '''
        Parametrization of electrons emitted by a pulse of the lamp.
        '''
        calc = 6e4*np.exp(-(t-2.8)**2/2/(2.90/2.355)**2 )
        return calc
    
    def emitted_electrons_in_interval(self,t0,tf, error = False):
        '''
        Integrate the lamp pulse to number of electrons.
        '''
        population, error = scipy.integrate.quad(Pulse_lamp, 0,6,epsrel = 1e-6)
        if error == False:
            return population
        else:
            return population, error
    
    def init_positions(self, n_electrons, shape):
        '''
        Initial spread of electrons on x,y and z.
        Standard version uses a Gaussian spread over x and y 
        with sigma given by the lamp aperture.
        '''
        sigma_xy = self.numerical_aperture * self.distance2photok
        mu, sigma = 0, np.sqrt(sigma_xy) # mean and standard deviation
        X0 = np.random.normal(mu, sigma, n_electrons)
        Y0 = np.random.normal(mu,sigma, n_electrons)
        #Z0 = np.random.normal(mu,1e-3, n_electrons)
        Z0 = np.zeros(n_electrons)
        return X0,Y0,Z0
    
class LCEPattern():
    '''
    Ad-hoc LCE maps. From pos and number of electrons to photons detected.
    '''
    
    def __init__(self, TPC):
        
        # TO BE CLEANED UP
        self.TPC = TPC
        self.z_gate = TPC.z_gate
        self.z_anode = TPC.z_anode
        self.gate_mesh = TPC.gate_mesh
        self.r_max = TPC.radius
        
        #self.load_patterns()
    
    def define_pattern_props(self, x_bin_step, y_bin_step, n_traces, smooth_pattern, force_traces = False):
        '''
        Call this function to define the pattern calculation properties.
        '''
        
        self.x_bin_step = 1 #mm #to put in config
        self.y_bin_step = 1 #mm #to put in config
        self.pattern_bin_area = self.x_bin_step * self.y_bin_step
        if force_traces != True:
            assert n_traces < 1e7, 'Asked for too many traces. Proceed on your own accord with force_traces = True'
        self.n_traces = int(n_traces)
        self.smooth_pattern = smooth_pattern
        return None
    
    #@classmethod
    def get_xy_on_plane(self,x0,y0,z0,directions,z_prime):
        '''
        Return the (x,y) of the intersection of a vector with direction 
        (theta,phi), physics convention, starting at point (x0,y0,z0) and
        finishing at the plane z=z_prime.
        '''
        x_prime = x0 + (z_prime-z0) * np.cos(directions[:,1]) * np.tan(directions[:,0])
        y_prime = y0 + (z_prime-z0) * np.sin(directions[:,1]) * np.tan(directions[:,0])
        final_pos = np.stack((x_prime, y_prime), axis = 1)
        return final_pos
    
    #@classmethod
    def get_xy_on_circ_array(self, final_pos):
        final_r = TPC.get_r(final_pos[:,0],final_pos[:,1])
        mask_r = final_r < self.r_max
        final_pos_array = final_pos[mask_r]
        return final_pos_array
    
    #@classmethod
    def get_hits_on_circ_array(self, x0,y0,z0):
        '''
        Get the positions of the toys that hit the circular area of the array.
        '''
        
        thetas = np.random.uniform(0,np.pi/2,self.n_traces)
        phis = np.random.uniform(0,2*np.pi,self.n_traces)
        directions = np.stack((thetas,phis), axis = 1)
        
        final_pos = self.get_xy_on_plane(x0,y0,z0,directions, self.z_anode)
        final_pos_array = self.get_xy_on_circ_array(final_pos)
        return final_pos_array
    
    #@classmethod
    def print_stats_of_hits(self, hits, n_traces):
        print('Initial number of photons: %s'%n_traces)
        print('Number of photons hit: '+
              '%s (%.2f%% of produced, %.2f of '+
              'full emission)'%(len(hits),
                                len(hits)/n_traces*100,
                                len(hits)/n_traces*100/2))
        return None
    
    #@classmethod
    def load_patterns(self,redo_patterns):
        '''
        Load the interpolated functions of the pattern for a set of points.
        Checks if they already exist and if `redo=True` to compute new ones
        '''
        pass
       
        if redo_patterns == True:
            #for each hex center:
            pattern = self.make_pattern_density(x_bin_step, y_bins_step)
            with open('patterns/hex_%d.pck', 'wb') as file:
                pickle.dump(pattern, file)
    
    #make this @classmethod??
    def get_pattern_density_hist2d(self, pos):
        
        x_min = y_min = -np.ceil(self.r_max*1.1)
        x_max = y_max = np.ceil(self.r_max*1.1)
        
        x_bin_sides = np.arange(x_min,x_max+self.x_bin_step*0.1,self.x_bin_step)
        y_bin_sides = np.arange(y_min,y_max+self.x_bin_step*0.1,self.y_bin_step)
        
        x_bin_middles = x_bin_sides[:-1] + self.x_bin_step/2
        y_bin_middles = y_bin_sides[:-1] + self.y_bin_step/2
        
        hist2d = np.histogram2d(pos[:,0],pos[:,1],
                        bins = (x_bin_sides,y_bin_sides),
                                density = False)
        #density could be True but we can also do it by hand given the bin area 
        #and total of photons. It's even better because it can be normalized properly 
        #from the start taking into account the photonos projected downwards and the
        #ones that miss the array.
        assert np.sum(hist2d[0]) == len(pos), ('Lost some photons on the histogram??\n'+
                                               'N_toys: %d\n' %len(pos)+
                                               'N_hist2d: %d' %len(hist2d[2]))
        
        
        #total_in_hist = np.sum(hist2d[0])
        #double the traces innormalization because of under side of sphere
        hist_density = hist2d[0]/self.pattern_bin_area/(self.n_traces*2) #fraction/mm^2;
        return x_bin_middles,y_bin_middles,hist_density
    
    # make @classmethod ??
    def make_pattern_density(self,pos):
        '''
        Takes the toy-MC results and makes the 2D density histogram, 
        normalized to the total number of traces produced and bin area.
        Returns either the interpolative function.
        Params:
          * pos: (N,2) array of the positions of hits in the circular array
          * x_bin_step: size of the hist2d bin length in x
          * y_bin_step: size of the hist2d bin length in y
        '''
        if self.smooth_pattern:
            s = 0.0000001
        else:
            s = 0
        x_bin_middles,y_bin_middles,hist_density = self.get_pattern_density_hist2d(pos)
        interp2s = scipy.interpolate.RectBivariateSpline(x_bin_middles, 
                                                         y_bin_middles,
                                                         hist_density,
                                                         s = s)
        return interp2s
    

### MAIN ###

r_max, hex_size = 75, 1.56
length = 2600
liquid_gap = 5
gas_gap = 5
drift_field = 100

Xenoscope = TPC(r_max, length, liquid_gap, gas_gap, drift_field)
mesh = Xenoscope.gate_mesh

scintillation = LCEPattern(Xenoscope)
scintillation.define_pattern_props(x_bin_step = 1, y_bin_step = 1, 
                                   n_traces = 1e8, smooth_pattern = False)

def make_pattern_local(x0,y0,z0):
    toy_events = scintillation.get_hits_on_circ_array(x0,y0,z0)
    pattern = scintillation.make_pattern_density(toy_events)
    return pattern

for hex_id,hex_position in tqdm(enumerate(initial_toy_pos), total = mesh.n_hexes):
    pattern = make_pattern_local(hex_position[0],hex_position[1],hex_position[2])
    with open('patterns/hex_%d.pck'%hex_id, 'wb') as file:
        pickle.dump(pattern, file)