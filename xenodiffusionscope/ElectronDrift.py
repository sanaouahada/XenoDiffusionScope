import numpy as np
import warnings
from tqdm import tqdm
from .TPC import TPC

class ElectronDrift:
    '''
    Processes and variables related to teh drift of electrons through the LXe TPC.
    Needs the input of a TPC object and an int n_electrons
    '''
    
    def __init__(self, tpc, xelamp,drift_delta_t):
        self.tpc = tpc
        self.drift_velocity = tpc.drift_velocity 
        self.drift_delta_t = drift_delta_t # us
        
        self.sigma_trans = np.sqrt(2*tpc.diffusion_trans*self.drift_delta_t)
        self.sigma_long = np.sqrt(2*tpc.diffusion_long*self.drift_delta_t)
        
        self.lamp = xelamp
        
        #harcoded values to go to a config file
        self.e_lifetime = 2000 #us
        self.se_gain = 28.57 #pe/e- from Xurich ii
        self.extract_efficiency = 0.99 # Electron extraction efficiency
        
        warnings.formatwarning = TPC.simple_warning
    
    # def check_boundaries(self):
    #     '''
    #     !NOT IN USE!
    #     Check if all the electrons are within the boundaries.
    #     If outside r>r_max -> turn to nan
    #     If at z=0 (gate) -> put into X/Y/Z_gas
    #     '''
    #     mask_at_gate = Z >= 0
        
    #     t_gas[mask_at_gate] = self.t
        
    #     R = get_r(X,Y)
    #     mask_outside_bound = R > self.TPC.radius
    #     X[mask_outside_bound] = np.nan
    #     Y[mask_outside_bound] = np.nan
    #     Z[mask_outside_bound] = np.nan
    
    def drift_lamp_pulse_slice(self, t0_pulse_slice, tf_pulse_slice):
        '''
        Drift electrons from a given integration period of the lamp pulse.
        '''
        
        n_electrons = self.lamp.emitted_electrons_in_interval(t0_pulse_slice, tf_pulse_slice)
        x0,y0,z0 = self.lamp.init_positions(n_electrons)
        x,y,z = x0.copy(),y0.copy(),z0.copy()
        
        self.t_running = 0
        warnings.warn('Starting drifting process with dt=%.2f us. Light-speed!'%self.drift_delta_t)
        half_warning = False
        while np.any(z<self.tpc.length): 
            #there is a catch here that I'm not recording the x,y 
            #when they cross z=tpc.length but only when ALL cross z=tpc.length
            # TO BE FIXED
            x,y,z = self.drift_step(x,y,z)
            
            if np.any(z>self.tpc.length/2) and half_warning==False:
                warnings.warn('An electron reached half-way there in %d us.' %self.t_running)
                half_warning = True
                
        warnings.warn('All electrons have drifted to the gate after %d us.' %self.t_running)
        
        return x,y,z
    
    def drift_full_pulse(self, lamp_end_time = 6):
        delta_t_lamp = self.lamp.delta_t_lamp
        times_lamp = np.arange(0,8, self.lamp.delta_t_lamp)
        
        positions_lamp_slices = dict() # bulky but practicle
        
        for t_lamp_index in tqdm(range(len(times_lamp) - 1)):
            x,y,z = self.drift_lamp_pulse_slice(times_lamp[t_lamp_index],
                                                times_lamp[t_lamp_index + 1])
            positions_lamp_slices['slice_%d'%t_lamp_index] = dict(start = times_lamp[t_lamp_index],
                                                                  end = times_lamp[t_lamp_index + 1],
                                                                  delta_t_lamp = delta_t_lamp,
                                                                  x = x,
                                                                  y = y,
                                                                  z = z)
        return positions_lamp_slices
    
    def drift_step(self, x,y,z):
        '''
        Make an increment on the electron array position following diffusion.
        Currently the arrival times are not recorded.
        '''
        _n_electrons = len(x)
        delta_x = np.random.normal(0, self.sigma_trans, _n_electrons)
        delta_y = np.random.normal(0, self.sigma_trans, _n_electrons)
        delta_z = (np.random.normal(0, self.sigma_long, _n_electrons) +
                   self.drift_velocity * self.drift_delta_t)
        
        x = x + delta_x
        y = y + delta_y
        z = z + delta_z
        
        self.t_running += self.drift_delta_t
        return x,y,z

    def apply_corrections(self, x,y,z):
        '''
        Apply all the reductions needed.
        '''
        x,y,z = self.apply_elifetime(x,y,z)
        x,y,z = self.extract_electrons(x,y,z)
        
        return x,y,z
    
    def extract_electrons(self,x,y,z):
        '''
        Apply extraction efficiency.
        '''
        n_electrons = len(x)
        n_unlucky = round(n_electrons * (1-self.extract_efficiency))
        unlucky_electrons_index = np.random.choice(n_electrons,size = n_unlucky, replace=False)  
        
        x = np.delete(x, unlucky_electrons_index)
        y = np.delete(y, unlucky_electrons_index)
        z = np.delete(z, unlucky_electrons_index)
        # #turn to nan the unlucky electrons
        # x[unlucky_electrons_index] = np.nan
        # y[unlucky_electrons_index] = np.nan
        # z[unlucky_electrons_index] = np.nan
        
        return x,y,z
    
    def apply_elifetime(self,x,y,z):
        '''
        Reduce the ammount of electrons due to the non-infinite e-lifetime.
        To finish: x,y,z_old are the final arrays of electrons.
        ''' 
        n_electrons = len(x)
        drift_time = self.tpc.length / self.drift_velocity # should be electron dependant in the future
       
        n_unlucky = round(n_electrons *(1-np.exp(-drift_time/self.e_lifetime)))
        unlucky_electrons_index = np.random.choice(n_electrons,size = n_unlucky, replace=False)  
        
        x = np.delete(x, unlucky_electrons_index)
        y = np.delete(y, unlucky_electrons_index)
        z = np.delete(z, unlucky_electrons_index)
        
        # #turn to nan the unlucky electrons
        # x[unlucky_electrons_index] = np.nan
        # y[unlucky_electrons_index] = np.nan
        # z[unlucky_electrons_index] = np.nan
        
        return x,y,z
    
    def convert_electron_to_photons(self,n_electrons):
        '''
        This will be a very fancy method to infer the number of either photons or 
        pe or whatever. For now it uses the SE gain value of either Xurich II.
        Please make a PR.
        '''
        
        n_pe = n_electrons * self.se_gain
        
        return n_pe
    