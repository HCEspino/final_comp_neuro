import networkx as nx
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage

def get_neighbors(x, y, MAPSIZEX, MAPSIZEY):
    nl = []
    left = False
    right = False
    up = False
    down = False
    
    if x - 1 >= 0:
        nl.append([x-1, y])
        left = True
    if x + 1 < MAPSIZEX:
        nl.append([x+1, y])
        right = True
    if y - 1 >= 0:
        nl.append([x, y-1])
        down = True
    if y + 1 < MAPSIZEY:
        nl.append([x, y+1])
        up = True
        
    if up and right:
        nl.append([x+1, y+1])
    if down and right:
        nl.append([x+1, y-1])
    if up and left:
        nl.append([x-1, y+1])
    if down and left:
        nl.append([x-1, y-1])
        
    return nl

def norm_scale(minval, maxval, scale, X):
    return ((1 - (X - minval) / (maxval - minval)) * scale) + 3

def draw_weights(MAPSIZEX, MAPSIZEY, XSCALE, YSCALE, CSMIN, CSMAX, title, wgt):
    avg = []
    minwt = None
    maxwt = None

    for x in range(MAPSIZEX):
        row = []
        for y in range(MAPSIZEY):
            wpsum = 0
            neighbors = get_neighbors(x, y, MAPSIZEX, MAPSIZEY)
            for i in neighbors:
                nw = wgt[i[0]][i[1]][x][y]
                if minwt == None or nw < minwt:
                    minwt = nw
                if maxwt == None or nw > maxwt:
                    maxwt = nw
                wpsum += nw
            row.append(wpsum / len(neighbors))
        avg.append(row)

    G = nx.DiGraph()

    for y in range(0, -MAPSIZEY, -1):
        for x in range(MAPSIZEX):
            G.add_node(f"{x}, {y*-1}",pos=(x*XSCALE,y*YSCALE), weight=avg[x][y*-1])
            for i in get_neighbors(x, y*-1, MAPSIZEX, MAPSIZEY):
                G.add_edge(f"{x}, {y*-1}", f"{i[0]}, {i[1]}", weight=wgt[x][y*-1][i[0]][i[1]])
                
    edges = G.edges()
    edge_colors = [G[u][v]['weight'] for u,v in edges]
    edge_size = [norm_scale(minwt, maxwt, 6, G[u][v]['weight']) for u,v in edges]
    cmap = plt.cm.inferno

    weights = G.edges(data="weight")
    wtlst = []
    for i in weights:
        wtlst.append(i[2])

    maxwt = max(wtlst)
    minwt = min(wtlst)

    pos = nx.get_node_attributes(G,'pos')

    color_data = list(G.nodes(data="weight", default=1))
    node_colors = []
    for i in color_data:
        node_colors.append(i[1])

    plt.figure(figsize=(34,30))

    nodes = nx.draw_networkx_nodes(
        G, 
        pos, 
        node_size=1200, 
        node_color=node_colors, 
        cmap = cmap, 
        vmin=CSMIN, 
        vmax=CSMAX)
    edges = nx.draw_networkx_edges(
        G,
        pos,
        node_size=1200,
        arrowstyle="->",
        arrowsize=15,
        edge_color=edge_colors,
        edge_cmap=cmap,
        edge_vmin=CSMIN,
        edge_vmax=CSMAX,
        width=edge_size,
        connectionstyle='arc3, rad=0.1')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = CSMIN, vmax=CSMAX))
    sm._A = []
    edgebar = plt.colorbar(sm)
    edgebar.ax.set_ylabel("Edge Cost", fontsize=18)

    plt.title(f"Cost map")
    plt.savefig(f"{title}.png", dpi=300, bbox_inches='tight')