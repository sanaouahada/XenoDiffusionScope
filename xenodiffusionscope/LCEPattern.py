import numpy as np
import scipy.integrate
import scipy.interpolate
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from .TPC import TPC

#Matplotlib plots should be handled in separate file with Plotter.

class LCEPattern:
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
        
        self.x_bin_step = x_bin_step #mm #to put in config
        self.y_bin_step = y_bin_step #mm #to put in config
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
    #def load_patterns(self,redo_patterns):
    #    '''
    #    Load the interpolated functions of the pattern for a set of points.
    #    Checks if they already exist and if `redo=True` to compute new ones
    #    '''
    #    pass
    #   
    #    if redo_patterns == True:
    #        #for each hex center:
    #        pattern = self.make_pattern_density(x_bin_step, y_bins_step)
    #        with open('patterns/hex_%d.pck', 'wb') as file:
    #            pickle.dump(pattern, file)
    
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
    
    @staticmethod
    def plot_pattern(tpc,pattern, hex_id):
        fig,ax = plt.subplots(1,1,figsize = (9,9), dpi = 100)
        ax.set_title('Pattern interpolation\n(spline, k=3)')
        _x = np.arange(-80,80,1)
        _y = np.arange(-80,80,1)
        _xx,_yy = np.meshgrid(_x,_y, indexing='ij')
        _rr = TPC.get_r(_xx,_yy)
        _xx = _xx[_rr < tpc.r_max]
        _yy = _yy[_rr < tpc.r_max]
        _zz = pattern.ev(_xx,_yy)
        interpolated = ax.scatter(_xx, _yy, c=np.log10(_zz), marker = 's',
                                  s = 3, vmin = -6.2)

        ax.add_patch(Circle((0,0),75, color = 'r',fill = False, linewidth = 1, ls ='--'))
        ax.set_aspect('equal')
        fig.colorbar(interpolated, ax = ax)

        plt.savefig('figures/patterns/hex_v0_%d' %hex_id)
