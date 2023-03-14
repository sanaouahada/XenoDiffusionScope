import numpy as np
from hexalattice.hexalattice import create_hex_grid
from tqdm import tqdm


class MeshGrid:
    '''
    Grid-related stuff. Construct, change, focus, etc.
    These helpstrings are becoming worse and worse, I know.
    '''
    
    def __init__(self, r_max, hex_side):
        self.r_max = r_max
        self.hex_side = hex_side
        self.hex_short_diagonal = self.hex_side * np.sqrt(3)
        self.hex_long_diagonal = 2 * self.hex_side
        
        self.hex_centers = self.construct_mesh()
        self.n_hexes = len(self.hex_centers)
        
    def construct_mesh(self):
        '''
        Construct the hexagonal gate mesh middles.
          * r_max: radius of the electrode
          * a: size of the hexagons
        '''
        
        n_hex_x = 2 * np.ceil(self.r_max/self.hex_short_diagonal)
        n_hex_y = 2 * np.ceil(self.r_max/(1.5*self.hex_side))
        
        hex_centers, _ = create_hex_grid(nx=n_hex_x, ny=n_hex_y, 
                                         min_diam = self.hex_short_diagonal,
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
    
    def count_focused(self, pos):
        '''
        Counts the number of events on each hex center.
        '''
        N_counts = np.zeros(self.n_hexes)
        for _hex_i, _hex_xy in tqdm(enumerate(self.hex_centers),
                                    'Counting hits in hex centers',
                                    total = self.n_hexes):
            N_counts[_hex_i] = len(np.where((pos[:,0] == _hex_xy[0]) &
                                            (pos[:,1] == _hex_xy[1])
                                           )[0])
        assert np.sum(N_counts) == len(pos), 'Lots some electrons counting. Woops!'
        return N_counts
