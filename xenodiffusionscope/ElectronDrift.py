import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.font_manager as fm

from .TPC import TPC

import warnings

class ElectronDrift:

    '''
    Process and models related to the drift of electrons through the LXe TPC. 
    input : initial positions of electron cloud 
    '''
    def __init__(self, initial_positions, tpc):
        # Initialise with electron cloud intiial positions and TPC 
        self.x, self.y, self.z = initial_positions[:, 0], initial_positions[:, 1], initial_positions[:, 2]
        self.tpc = tpc
        self.drift_velocity = tpc.drift_velocity #mm/us

        # Harcoded values to go to a config file
        self.e_lifetime = 2000 #us
        self.se_gain = 28.57 #pe/e- from Xurich ii
        self.extract_efficiency = 0.99 # Electron extraction efficiency

        # Variables to keep track of the simulation
        self.t_running = 0
    
    def drift_step(self):
        dt = 1.0  # Time step for the drift simulation in microseconds
        # Calculate drift velocity, longitudinal diffusion, and transverse diffusion
        drift_velocity = self.tpc.drift_velocity
        diffusion_long = self.tpc.diffusion_long
        diffusion_trans = self.tpc.diffusion_trans
        
        # Update z-coordinate based on the drift velocity
        self.z += drift_velocity * dt
        
        # Update x and y coordinates with longitudinal and transverse diffusion
        self.x += np.random.normal(0, np.sqrt(2 * diffusion_long * dt), len(self.x))
        self.y += np.random.normal(0, np.sqrt(2 * diffusion_trans * dt), len(self.y))
        
        self.t_running += dt
    
    def drift_electrons(self):
        warnings.formatwarning = self.tpc.simple_warning
        warnings.warn('Starting drifting process with dt=%.2f us.' % 0.1)
        
        half_warning = False
        while np.any(self.z < self.tpc.length):
            self.drift_step()
            
            if np.any(self.z > self.tpc.length / 2) and not half_warning:
                warnings.warn('An electron reached halfway there in %d us.' % self.t_running)
                half_warning = True
        
        warnings.warn('All electrons have drifted to the gate after %d us.' % self.t_running)
        return self.x, self.y, self.z
    
    def apply_corrections(self, x,y,z):
        '''
        Apply all the reductions needed.
        '''
        x,y,z = self.apply_elifetime(x,y,z)
        x,y,z = self.extract_electrons(x,y,z)
        
        # Restrict x and y coordinates to TPC volume
        x = np.clip(x, -75, 75)
        y = np.clip(y, -75, 75)
    
        return x,y,z
    
    def extract_electrons(self,x,y,z):
        '''
        Apply extraction efficiency.
        '''
        n = len(x)
        n_unlucky = round(n * (1-self.extract_efficiency))
       
        unlucky_electrons_index = np.random.choice(n, size = n_unlucky, replace=False)  
       
        x = np.delete(x, unlucky_electrons_index)
        y = np.delete(y, unlucky_electrons_index)
        z = np.delete(z, unlucky_electrons_index)
       
        return x,y,z
    
    def apply_elifetime(self,x,y,z):
        '''
        Reduce the ammount of electrons due to the non-infinite e-lifetime.
        ''' 
        n = len(x)
        drift_time = self.tpc.length / self.drift_velocity # should be electron dependant in the future
       
        n_unlucky = round(n *(1-np.exp(-drift_time/self.e_lifetime)))
        n_unlucky = int(n_unlucky)  # Convert to integer
        unlucky_electrons_index = np.random.choice(n,size = n_unlucky, replace=False)  
        
        x = np.delete(x, unlucky_electrons_index)
        y = np.delete(y, unlucky_electrons_index)
        z = np.delete(z, unlucky_electrons_index)
        
        return x,y,z
    
    def convert_electron_to_photons(self,n_electrons):
        '''
        This will be a very fancy method to infer the number of either photons or 
        pe or whatever. For now it uses the SE gain value of either Xurich II.
        Please make a PR.
        '''
        
        n_pe = n_electrons * self.se_gain
        
        return n_pe
    

