import numpy as np
import warnings


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
        
        self.drift_velocity = self.model_velocity(self.drift_field) #mm/us
        self.diffusion_long = self.model_diffusion_longitudinal(self.drift_field) #mm2/us
        self.diffusion_trans = self.model_diffusion_transversal(self.drift_field) #mm2/us
        
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
        x = r*np.cos(phi)*sin(theta)
        y = r*np.sin(phi)*sin(theta)
        z = r*np.cos(theta)
        return x,y,z
    
    @staticmethod
    def simple_warning(message, category, filename, lineno, file=None, line=None):
        return 'WARNING: %s\n' %message
    
    