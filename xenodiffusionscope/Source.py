import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import nestpy

class Source:
    '''
    Input file: energy and positions of interactions between gamma particle - LXe atoms.
    NEST code generates yields of scintillation photons and primary electrons. 
    '''
    def __init__(self, filename):
        """"
        Initialise source with arrays of energy and positions.
        """
        self.df = pd.read_csv(filename, header=0) 

        # Access the required columns 
        self.x_position = self.df['x_position']
        self.y_position = self.df['y_position']
        self.z_position = self.df['z_position']
        self.energy_dep = self.df['energy_dep']*1e6 #Scale the energy values from GeV to keV 


        # Calculate min and max energy
        self.min_energy = self.energy_dep.min()
        self.max_energy = self.energy_dep.max()
        
    def read_energies(self): 
        energies = np.array(self.energy_dep)
        return energies
    
    def n_initial_electrons(self, interaction = nestpy.nr, density = 2.9, field = 100.):
        """
        Get yields from NEST for specific interaction, energy and drift field
        """
        # Initialise nest calculator to call functions 
        nc = nestpy.NESTcalc(nestpy.VDetector())
        
        # List to store the number of electrons for each energy
        n_electrons = []  
    
        # Iterate over the energy values
        for E in self.energy_dep:
            yields = nc.GetYields(interaction, E, density, field)
            n_electron = int(yields.ElectronYield)
            
            # Append the number of electrons to the list
            n_electrons.append(n_electron)  

        return n_electrons
    
    def pos_initial_electrons(self, n_electrons):
        '''
        Creates electron cloud assimilated to a point source at each interaction vertex. Needs input of n_electrons array:      number of electrons generated at each vertex. 
        '''
        electron_sources = []  # List to store the generated electron sources

        # Iterate over the indices of the arrays
        for i in range(len(n_electrons)):
            n = n_electrons[i]  # Electron count at current index
            x = self.x_position[i]  # x position at current index
            y = self.y_position[i]  # y position at current index
            z = self.z_position[i]  # z position at current index
            #print(n,x,y,z)

            # Create a point source of electrons at the given position with the specified count
            electron_source = [(x, y, z)] * n

            # Add the electron source to the list
            electron_sources.extend(electron_source)

        # Flatten the electron_sources list and store the positions in a separate array
        positions = np.array(electron_sources)
        
        return positions
    


