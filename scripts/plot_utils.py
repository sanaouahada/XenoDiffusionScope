import numpy as np
import os 
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.colors import LinearSegmentedColormap

# Set the fontsize for all elements
plt.rcParams.update({'font.size': 12})

from my_colors import sunset as colors 

# Define positions for custom heatmap
positions = np.linspace(0, 1, len(colors))  # Positions along the colormap

# Create the custom colormap using LinearSegmentedColormap
custom_cmap = LinearSegmentedColormap.from_list("Custom Colormap", list(zip(positions, colors)))

def electrons_cloud_plot(positions, title):
        # Calculate 2D histogram
        bins = 200
        hist, xedges, yedges = np.histogram2d(positions[:,0], positions[:,1], bins=bins)

        # Plot 2D histogram
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')

        ax.set_title(title) 
        im = ax.imshow(hist.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap=custom_cmap)
        fig.colorbar(im)

def electrons_scatter_plot(x, y, title):
        # Create a scatter plot for x and y positions
        plt.scatter(x, y, alpha=0.5, s=20, c=colors[2])  ##add label = 'n electrons'
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')

        plt.xlim([-80, 80])
        plt.ylim([-120, 60])

        plt.title(title)
        plt.legend(loc='upper left')
        plt.tight_layout()
        

## Focusing on mesh grid
def focusing(x1, y1, x2, y2, title):
        '''
        Electron positions before and after focusing on the hex centers of a mesh grid. 
        '''
        plt.figure(figsize=(6, 6))
        plt.scatter(x1, y1, label='Unfocused', alpha=0.5, s=20, c=colors[1])
        plt.scatter(x2, y2, label='Focused', marker='x', linewidth = 1.1, c=colors[10], s=25)
        plt.gca().set_aspect('equal', fontsize=14)
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        plt.title(title, fontsize=14)
        plt.legend(loc='upper left')
        plt.tight_layout()

## Number of electrons histogram
def electrons_hist(x, y, title):
        plt.figure(figsize=(6, 4))
        plt.bar(np.arange(0, x, 1), y, color='#4A7BB7', alpha=0.8, linewidth = 1.5)
        plt.xlabel('Hex number')
        plt.ylabel('# of electrons')
        plt.yscale('log')
        plt.title(title, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
   

