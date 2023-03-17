import numpy as np
import warnings

from .MeshGrid import MeshGrid

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
        Longitudinal diffusion model from NEST v2.0.0 .
        '''
        ans = (57.381 * np.power(drift_field, -0.22221) + 
               127.27 * np.exp(-drift_field /32.821))
        
        return ans*1e-4 #mm2/us
    
    @classmethod
    def model_diffusion_transversal(self, drift_field):
        '''
        Transversal diffusion model from NEST v2.0.0 .
        '''
        ans = (37.368 * np.power(drift_field, .093452) * 
               np.exp(-8.1651e-5 * drift_field))
        
        return ans*1e-4 #mm2/us
    
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
    
    @staticmethod
    def simple_warning(message, category, filename, lineno, file=None, line=None):
        return 'WARNING: %s\n' %message
    
    