import numpy as np
import warnings
import scipy.integrate
import scipy.interpolate


class XeLamp:
    '''
    Properties and functions of the Xe lamp and its brightly 
    shining effects on producing electrons on the photocathode.
    '''
    
    def __init__(self,delta_t_lamp):
        self.numerical_aperture = 0.22
        self.distance2photok = 2 # mm
        self.length_xy = self.numerical_aperture*self.distance2photok
        
        #time step to consider when reconstructing the lamp pulse
        #use 2 ns (0.002 us) to match ADC freq or 0.25 for testing 
        #(from YB's original code)
        self.delta_t_lamp = delta_t_lamp 
        self.times_lamp = np.arange(0,6,self.delta_t_lamp)
        
    @classmethod
    def pulse_lamp(cls,t):
        '''
        Parametrization of electrons emitted by a pulse of the lamp.
        '''
        calc = 6e4*np.exp(-(t-2.8)**2/2/(2.90/2.355)**2 )
        return calc
    
    def emitted_electrons_in_interval(self,t0,tf, error = False):
        '''
        Integrate the lamp pulse to number of electrons from t0 to tf. Gives population. 
        '''
        integral = scipy.integrate.quad(XeLamp.pulse_lamp,t0,tf,epsrel = 1e-6)
        
        if error==True:
            
            return int(integral[0]),integral[1]
        else:
            return int(integral[0])
        
    
    def init_positions(self, n_electrons, shape = 'circle'):
        '''
        Initial spread of electrons on x,y and z.
        Standard version uses a Gaussian spread over x and y 
        with sigma given by the lamp aperture.
        '''
        if shape != 'circle':
            warnings.warn('Not implemented yet. Taking circle.')
            
        sigma_xy = self.numerical_aperture * self.distance2photok
        mu, sigma = 0, np.sqrt(sigma_xy) # mean and standard deviation
        X0 = np.random.normal(mu, sigma, n_electrons)
        Y0 = np.random.normal(mu,sigma, n_electrons)
        #Z0 = np.random.normal(mu,1e-3, n_electrons)
        Z0 = np.zeros(n_electrons)
        return X0,Y0,Z0