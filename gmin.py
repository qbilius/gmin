#!/usr/bin/env python

# gmin: a minimal model with grouping principles
# Copyright 2012-2013 Jonas Kubilius
# The program is distributed under the terms of the GNU General Public License,
# either version 3 of the License, or (at your option) any later version.

"""
gmin: a minimal model with grouping principles
"""

import sys
import cPickle as pickle
import numpy as np
import scipy.ndimage
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from scipy.sparse import csr_matrix


class GMin(object):

    def __init__(self):
        self.oris = np.pi/16*np.arange(16)
        #self.sfs = np.arange(7,39,4)
        #self.sfs = 256/5/self.sfs
        #self.sfs = np.logspace(np.log(0.1), np.log(50), num=8, base=np.e)
        #self.sfs = np.logspace(np.log(1), np.log(5), num=8, base=np.e)
        self.sfs = np.linspace(1, 8, num=8)

    def contrast_sens(self):
        # HPmH model from Watson & Ahumada (2005, doi: 10.1167/5.9.6)
        gain = 373.08
        y = gain * (1 / np.cosh((self.sfs / 4.1726)**0.7786) - \
             .8493 / np.cosh(self.sfs / 1.3625))
        return y

    def gabor(self, sf=42, theta=0, gamma=1, phase=0):
        """
        Returns a Gabor filter as in Mutch and Lowe, 2006.

        :Kwargs:
            - sf (int or float, default:42)
                Spatial frequency (FIXME: it's probably 1/sf)
            - theta (float, default: 0)
                Gabor orientation
            - gamma (float, default: 1)
                Skew of Gabor. By default the filter is circular, but
                manipulating gamma can make it ellipsoidal.
            - phase (float, default: 0)
                Phase of Gabor. 0 makes it an even filter,
                numpy.pi/2 makes it an odd filter.
        :Returns:
            A 2D numpy.array Gabor filter
        """
        #
        sigma = sf/6.
        lam = 2.8*sf/6.
        r = int(sigma*3)  # there is nothing interesting 3 STDs away
        theta -= np.pi/2  # we want 0 deg to be oriented along the x-axis

        x,y = np.meshgrid(np.arange(-r,r+1),np.arange(r,-r-1,-1))
        # rotate coordinates the opposite way than Gabor
        X = x*np.cos(-theta) - y*np.sin(-theta)
        Y = x*np.sin(-theta) + y*np.cos(-theta)
        g = np.exp( -(X**2 + (gamma*Y)**2) / (2*sigma**2) ) * np.cos( 2*np.pi*X/lam + phase)

        g -= np.mean(g)  # mean 0
        #g /= np.sum(g**2) # energy 1
        g /= np.max(np.abs(g))
        #g[np.abs(g)<.001] = 0  # remove spillover
        #g -= np.min(g)  # all positive

        return g

    def plot_gabor_oris(self):
        gabors = [self.gabor(theta=ori) for ori in self.oris]
        gabors = np.hstack(gabors)
        return gabors

    def plot_gabor_sizes(self, stim):
        gsize = int(stim.shape[0]/stim.size_deg/np.min(self.sfs)) + 1
        gabim = []
        for p in range(4):
            gabim2 = []
            for sfno, sf in enumerate(self.sfs):
                g = self.gabor(sf=stim.shape[0]/stim.size_deg/sf,
                                   phase=p*np.pi/2)
                gabor = np.ones((gsize, gsize)) * g[0,0]
                gabor[(gsize - g.shape[0]) / 2: (gsize + g.shape[0]) / 2,
                      (gsize - g.shape[1]) / 2: (gsize + g.shape[1]) / 2] = g
                gabim2.append(gabor)
            gabim2 = np.hstack(gabim2)
            gabim.append(gabim2)
        gabim = np.vstack(gabim)

        if gabim.shape[0] < stim.shape[0]:
            gabim = np.vstack((gabim, np.zeros((stim.shape[0] - gabim.shape[0],
                    gabim.shape[1]))))
        if gabim.shape[1] < stim.shape[1]:
            gabim = np.hstack((gabim,
                     np.zeros((gabim.shape[0], stim.shape[1] - gabim.shape[1]))
                    ))
        return gabim

    def _get_rf_field(self, rf_shape):
        x,y = np.meshgrid(np.arange(-rf_shape[0]/2+1, rf_shape[0]/2+1),
                          np.arange(rf_shape[1]/2, -rf_shape[1]/2, -1))
        return x, y

    def proximity(self, rf_shape, r0=1.16):
        """
        Computes grouping by proximity.

        :Args:
            rf (2D numpy.array)
                Receptive field
        :Kwargs:
            r0 (float, default: 1.16)
                Spatial decay constant (in degrees visual angle). Default value
                taken from Ernst et al., Eq. 4 (doi:10.1371/journal.pcbi.1002520)
        :Returns:
            Probability of grouping
        """
        #import pdb; pdb.set_trace()
        #rf_sh = np.sqrt(rf.size)
        #rf = np.reshape(rf, (rf_sh, rf_sh))

        #rf_shape = rf.shape
        x, y = self._get_rf_field(rf_shape)
        r_dist = np.sqrt(x**2 + y**2)
        grouping = np.exp(-r_dist / (r0*self.deg2px))
        return grouping

    def similarity(self, rf):
        """Computes similarity for a given feature
        """
        # central element's value
        value = rf[rf.shape[0]/2, rf.shape[1]/2]
        grouping = np.exp(-(rf-value)**2)

        return grouping

    def good_cont(self, rf, r0=1.16, sigma_a=.25, sigma_b=.57):
        """
        Computes grouping by good continuation (association field).

        Implemented using the closed form solution from Ernst et al.
        (doi:10.1371/journal.pcbi.1002520)

        :Args:
            rf (2D numpy.array)
                Receptive field
        :Kwargs:
            r0 (float, default: 1.16)
                Spatial decay constant (in degrees visual angle). Default value
                taken from Ernst et al., Eq. 4 (doi:10.1371/journal.pcbi.1002520)
            sigma_a (float, default: .25)
                Tolerance to co-circularity; chosen to be optimal
                from doi:10.1371/journal.pcbi.1002520.g006
            sigma_b (float, default: .57)
                Curvature; chosen to be optimal
                from doi:10.1371/journal.pcbi.1002520.g007
        """
        x, y = self._get_rf_field(rf.shape)
        # central element's orientation
        theta = rf[rf.shape[0]/2, rf.shape[1]/2]
        # rotate coordinates the opposite way than assoc. field
        X = x*np.cos(-theta) - y*np.sin(-theta)
        Y = x*np.sin(-theta) + y*np.cos(-theta)
        x = X
        y = Y
        alpha = np.arctan2(y,x)  # angle of a line connecting the two centers
        #r_dist = np.sqrt(x**2+y**2)  # distance from the central element
        beta = rf  # other elements' orientations

        #Ad = np.exp(-r_dist/r0)
        At = np.cosh( np.cos((beta-2*alpha-theta)/2.)/sigma_a**2 + 4*np.cos((beta-theta)/2.)/sigma_b**2 )+\
             np.cosh( -np.cos((beta-2*alpha-theta)/2.)/sigma_a**2 + 4*np.cos((beta-theta)/2.)/sigma_b**2 )+\
             np.cosh( np.cos((beta-np.pi-2*alpha-theta)/2.)/sigma_a**2 + 4*np.cos((beta-np.pi-theta)/2.)/sigma_b**2 )+\
             np.cosh( -np.cos((beta-np.pi-2*alpha-theta)/2.)/sigma_a**2 + 4*np.cos((beta-np.pi-theta)/2.)/sigma_b**2 )

        #A = Ad*At
        A = At
        #c0 = 1/(4*np.pi**2*scipy.special.iv(0,1/sigma_a**2)*scipy.special.iv(0,1/sigma_b**2))
        c0 = np.cosh(1/sigma_a**2+4/sigma_b**2)
        A /= c0

        return A

    def curvature(self, rf):
        x, y = self._get_rf_field(rf.shape)
        angle_field = np.arctan2(y,x)
        theta = rf[rf.shape[0]/2, rf.shape[1]/2]
        # determine elements with an angle greater than 'angle'
        above = np.logical_or(angle_field >= theta, angle_field + np.pi < theta)
        # elements with an angle smaller than 'angle'
        below = np.logical_not(above)
        # magnitude of the figure-ground signal
        #mag = np.sum(window_weight[above]) - np.sum(window_weight[below])
        ##
        #mag /= np.sum(window_weight[above]) + np.sum(window_weight[below])
        #fgmag[i,j] = np.abs(mag)*1
        ## orthogonal angle in the direction of a stronger signal
        #fgangle[i,j] = angle + np.sign(mag)*np.pi/2

        mag = rf[above] + rf[below]
        return mag


    def plot_assoc_field(self, stim=None):
        if stim is None:
            minarr = 255
        else:
            minarr = min((stim.shape[1]-1, stim.shape[0]-1))
        array = np.zeros((minarr, minarr))
        field_shape = array.shape  #[int(np.sqrt(len(array))), int(np.sqrt(len(array)))]
        x,y = np.meshgrid(np.arange(-field_shape[0]/2+1, field_shape[0]/2+1),
                          np.arange(field_shape[1]/2, -field_shape[1]/2, -1))
        theta = np.pi/4
        alpha = np.arctan2(y,x)
        array = 2*alpha - theta
        array[array.shape[0]/2, array.shape[1]/2] = theta
        af = self.good_cont(array)
        return af


    def plot_arrows(self, stim, weights, angles, ax):
        """Plots arrows: length edge_map[0], direction edge-map[1].

        Only edge_map[:][indices] are used for plotting.
        """
        thisCmap = mpl.cm.get_cmap('Paired')
        norm = mpl.colors.Normalize(0, 1)
        z = np.linspace(0, 1, 5)
        z = z[1:-1]
        colors = thisCmap(norm(z))
        for i,j in stim.indices:
            #ang = edge_map[1,i,j] % (2*np.pi) - np.pi
            #print '%d' %(edge_map[1,i,j]/np.pi*180),
            #print np.abs(edge_map[0,i,j]),
            #if weights[i,j] != 0:
                #import pdb; pdb.set_trace()
            if angles[i,j] % (2*np.pi) < np.pi: #ang > -np.pi/2 and ang < np.pi/2:
                color = colors[0]
            else:
                color = colors[1]
            ax.arrow( j, i,
                    np.cos(angles[i,j]) * weights[i,j],
                    -np.sin(angles[i,j]) * weights[i,j],
                    head_width=.5, head_length=1, fc=color, ec=color)
        #ax.set_xlim([0, stim.shape[1]-1])
        #ax.set_ylim([0, stim.shape[0]-1])
        #ax.set_aspect('equal')

    def imshow_arrows(self, array, x=None, y=None, weights=1, spacing=1, ax=None):
        if isinstance(weights,(int, float)):
            weights = np.ones(array.shape) * weights
        if x is None and y is None:
            x,y = self._get_rf_field(array)
        ii,jj = np.meshgrid(np.arange(2*spacing-1, array.shape[0], 2*spacing),
                        np.arange(2*spacing-1, array.shape[1], 2*spacing))
        if ax is None:
            ax = plt.subplot(111)
        for i,j in zip(ii.ravel(),jj.ravel()):
            if x[i,j]==0 and y[i,j] == 0:
                fc = 'r'  # central element is red
            else:
                fc = 'k'
            ax.arrow( x[i,j], y[i,j],
                    np.cos(array[i,j]) * weights[i,j] * spacing,
                    np.sin(array[i,j]) * weights[i,j] * spacing,
                    head_width=spacing/8., head_length=spacing/8.,
                    fc=fc, ec=fc, alpha=weights[i,j])
        ax.set_xlim([-array.shape[1]/2+1, array.shape[1]/2+1])
        ax.set_ylim([-array.shape[1]/2+1, array.shape[1]/2+1])
        ax.set_aspect('equal')

    def _isinside(self, smallbox, largebox, i, j):
        """Checks if a smaller box centered around (i,j) is completely inside
        a larger box
        """
        return (i - smallbox[0]/2 > 0 and i + smallbox[0]/2 + 1 < largebox[0] and
            j - smallbox[1]/2 > 0 and j + smallbox[1]/2 + 1 < largebox[1])

    def _boxinds(self, box, i, j):
        """Returns indices of all box elements centered around (i,j)
        """
        return np.s_[i-box[0]/2: i+box[0]/2+1, j-box[1]/2: j+box[1]/2+1]

    def compute_edges(self, stim, image, bins=5000, nelems=1000):
        all_edges = np.zeros((4, len(self.oris),  len(self.sfs)) + stim.shape)

        # Step 1: extract edges using Gabor filters
        for p in range(4):  # 4 phases
            for oi, ori in enumerate(self.oris):
                for sfno, sf in enumerate(self.sfs):
                    gabor = self.gabor(sf=self.deg2px/sf,
                                       theta=ori, phase=p*np.pi/2)
                    edges = scipy.ndimage.correlate(image, gabor)
                    all_edges[p, oi, sfno] = edges

        aa = all_edges*1.
        # Step 2: polarity invariance
        all_edges = np.abs(all_edges)

        # Step 3: average across spatial frequencies (possibly weighted)
        all_edges = np.average(all_edges, axis=2)#, weights=weights)

        # Step 4: choose maximally responding orientation for each location
        edges = np.reshape(all_edges, (-1, all_edges.shape[-2], all_edges.shape[-1]))
        #idx = np.argmax(edges)
        #import pdb; pdb.set_trace()
        stim.oris = np.take(self.oris, np.argmax(edges,0)%len(self.oris))
        stim.contrasts = np.max(edges, axis=0)

        # Step 5: normalize responses globally
        stim.contrasts /= np.max(stim.contrasts)


    def select_inds(self, values, nelems=1000, sel='random', size=3):
        """Choose only about nelems at top responding locations
        """
        sparse, inds = self._sparse_local_max(values, nelems=nelems, sel=sel,
            size=size)
        return inds

    #def sparse_local_max(self, responses, size=7, nelems=1000, nbins=5000,
                         #sel='random'):
        #sparse, inds = self._sparse_local_max(responses, size=size,
                        #nelems=nelems, nbins=nbins, sel=sel)
        #sparse2, inds2 = self._sparse_local_max(responses, size=3,
                        #nelems=nelems, nbins=nbins)
        #sparse += sparse2
        #import pdb; pdb.set_trace()
        #inds = inds.tolist() + inds2.tolist()
        #inds = set([(i,j) for i,j in inds])
        #inds = np.array(list(inds))

        #print "final sparse inds:", inds.shape

        return sparse, inds

    def _sparse_local_max(self, responses, size=3, nelems=1000, nbins=5000,
                            sel='random'):
        if isinstance(size, int):
            size = (size, size)

        f = (size[0]/2, size[1]/2)
        x,y = np.meshgrid(np.arange(f[0], responses.shape[0]-f[0], size[0]),
                          np.arange(f[1], responses.shape[1]-f[1], size[1]))
        inds = np.vstack((x.ravel(), y.ravel())).T
        print "initial sparse inds:", inds.shape
        sparse = np.zeros(responses.shape)

        for i,j in inds:
            s = np.s_[i-f[0]:i+f[0]+1, j-f[1]:j+f[1]+1]
            ind = np.argmax(responses[s])
            mi, mj = np.unravel_index(ind, size)
            maxwin = np.zeros(size)
            thismax = responses[s][mi,mj]
            maxwin[mi,mj] = thismax
            sparse[s] = maxwin

        do_random = False
        if sel == 'thres':
            hist, bin_edges = np.histogram(sparse.ravel(),bins=nbins)
            last = np.nonzero(np.cumsum(hist[::-1]) > nelems)[0][0]
            threshold = bin_edges[nbins-last-1]
            inds = np.where(sparse>threshold)
            inds = np.vstack(inds).T
            if len(inds) > 1.2*nelems:
                do_random = True
            print "final thres inds:", inds.shape, threshold
        if sel == 'random' or do_random:
            inds = np.where(sparse>0)
            inds = np.vstack(inds).T
            n = np.min((len(inds),nelems))
            inds_ind = np.random.randint(0, len(inds), n)
            inds = inds[inds_ind]
            print "final random inds:", inds.shape
            #import pdb; pdb.set_trace()



        #sp = sparse[inds[:,0], inds[:,1]]
        #sp = sparse
        #plt.hist(sp[sp>0].ravel(), bins=100)
        #plt.axvline(threshold, color='r')
        #plt.show()

        #import pdb; pdb.set_trace()

        return sparse, inds

    def detect_global_maxima(self, responses, bins=5000, nelems=1000):
        hist, bin_edges = np.histogram(responses.ravel(),bins=bins)
        last = np.nonzero(np.cumsum(hist[::-1]) > nelems)[0][0]
        threshold = bin_edges[bins-last-1]
        inds = responses>threshold
        #print '# elements used: %d' % np.sum(inds)
        detected_maxima = inds#np.logical_not(inds)
        #response[np.logical_not(inds)] = 0
        indices = np.transpose(np.nonzero(inds))
        return detected_maxima, indices

    def detect_local_maxima(self, arr, size=3):
        # http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
        """
        Takes an array and detects the troughs using the local maximum filter.
        Returns a boolean mask of the troughs (i.e. 1 when
        the pixel's value is the neighborhood maximum, 0 otherwise)
        """
        import scipy.ndimage.filters as filters
        import scipy.ndimage.morphology as morphology
        if isinstance(size, int):
            size = (size, size)
        # define an connected neighborhood
        # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
        neighborhood = morphology.generate_binary_structure(len(arr.shape),2)
        neighborhood = np.ones(size)
        #import pdb; pdb.set_trace()
        # apply the local minimum filter; all locations of minimum value
        # in their neighborhood are set to 1
        # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
        local_max = (filters.maximum_filter(arr, footprint=neighborhood) == arr)
        # local_min is a mask that contains the peaks we are
        # looking for, but also the background.
        # In order to isolate the peaks we must remove the background from the mask.
        #
        # we create the mask of the background
        background = (arr==0)
        #import pdb; pdb.set_trace()
        #
        # a little technicality: we must erode the background in order to
        # successfully subtract it from local_min, otherwise a line will
        # appear along the background border (artifact of the local minimum filter)
        # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
        eroded_background = morphology.binary_erosion(
            background, structure=neighborhood, border_value=1)
        #
        # we obtain the final mask, containing only peaks,
        # by removing the background from the local_min mask
        detected_maxima = local_max - eroded_background
        #plt.imshow(detected_maxima);plt.show()
        #import pdb; pdb.set_trace()
        inds = np.where(detected_maxima)
        return detected_maxima, np.vstack(inds).T

    def compute_grouping(self, stim, affinity=None):
        window_shape = (int(stim.deg2px), int(stim.deg2px))

        nind = len(stim.indices)
        flatind = stim.indices[:,0]*nind+stim.indices[:,1]
        xy2x = dict(np.vstack([flatind, np.arange(nind)]).T)

        inds = np.indices(stim.shape)
        sel = ([inds[0,stim.indices[:,0],stim.indices[:,1]],
                inds[1,stim.indices[:,0],stim.indices[:,1]]])
        selfields = np.zeros(stim.shape, dtype=bool)
        selfields[sel[0], sel[1]] = True

        probs = np.zeros(stim.shape)
        group = np.zeros((nind, nind))

        #probs = scipy.ndimage.generic_filter(stim.contrasts,
                #eval('self.'+affinity), size=window_shape)
        #sys.exit()

        for i,j in stim.indices:
            if self._isinside(window_shape, stim.shape, i, j):
                s = self._boxinds(window_shape, i, j)
                # get their strengths
                window_weight = stim.contrasts[s]
                window_angle = stim.oris[s]  # element oris in that window

                if affinity == 'good_cont':
                    prob = self.good_cont(window_angle)
                elif affinity == 'similarity_contrasts':
                    prob = self.similarity(window_weight)
                elif affinity == 'similarity_ori':
                    prob = self.similarity(window_angle)
                elif affinity == 'proximity':
                    prob = self.proximity(window_shape)
                    #plt.imshow(prob); plt.show()
                else:
                    prob = window_weight
                    #raise Exception('grouping affinity %s not known' %affinity)

                probs[s] += prob
                weights_sel = selfields[s]
                curr_idx = (inds[0][s][weights_sel], inds[1][s][weights_sel])
                curr_idx = curr_idx[0]*nind + curr_idx[1]

                wslice = np.array([xy2x[c] for c in curr_idx])
                group[xy2x[i*nind+j], wslice] = prob[weights_sel].ravel()
                #import pdb; pdb.set_trace()

        #group = csr_matrix(group)
        probs /= window_shape[0] * window_shape[1]

        return probs, group

    def hcluster(self, stim):
        #from hcluster import pdist, linkage, dendrogram
        import hcluster
        iu = np.triu_indices(len(stim.group), 1)
        #
        Z = hcluster.linkage(stim.group[iu], 'single', 'ward')
        import pdb; pdb.set_trace()
        thres = Z[-2, 2]
        dend = hcluster.dendrogram(Z, color_threshold=thres)
        plt.show()
        clusters = self.get_clusters(Z, n_clusters=4)#thres=thres)
        colors = self.get_colors(len(clusters))
        #import pdb; pdb.set_trace()
        for cluster, color in zip(clusters, colors):
            sel = stim.indices[np.array(cluster)]
            plt.plot(sel[:,1], sel[:,0],'o',  color=color, )
        plt.show()

    def _affcluster(self, stim, radius=10):
        from sklearn.metrics.pairwise import euclidean_distances
        #from scipy.spatial import ConvexHull

        inds = stim.indices
        X = euclidean_distances(inds, inds)

        use_inds = range(len(inds))
        np.random.shuffle(use_inds)
        i = use_inds.pop()
        X[:,i] = np.inf
        clusters = []
        cluster = [i]

        while len(use_inds) > 0:

            sel = X[np.array(cluster)]
            mn = np.min(sel)
            if mn < radius:
                j = np.argmin(sel)
                j = np.unravel_index(j, sel.shape)[-1]
                del use_inds[use_inds.index(j)]
                X[:,j] = np.inf
                cluster.append(j)
                i = j
            else:
                clusters.append(cluster)
                np.random.shuffle(use_inds)
                i = use_inds.pop()
                X[:,i] = np.inf
                cluster = [i]

        labels = np.zeros(len(inds))
        for k, cluster in enumerate(clusters):
            #if len(cluster) > 100:
                labels[np.array(cluster)] = k
            #else:
                #labels[np.array(cluster)] = -1
        return labels

    def affcluster(self, stim, sim, ax=None, radius=6, n_clusters=2, niter=1,
        simthres=0.00112665583835):
        #start = True
        lablen = []
        labels = []
        #labels = [self._affcluster(stim, radius=12.5)]
        #import pdb; pdb.set_trace()

        if True:

            hist, bin_edges = np.histogram(sim.ravel(), bins=50)
            bin_edges = bin_edges[:-1]
            mean_stds = []
            last_obj_len = 0
            #last_labs = None

            for edges in bin_edges:
                labs, simils = self._simcluster(stim, sim, simthres=edges)
                stds = []
                #import pdb; pdb.set_trace()
                lablen = np.bincount(labs.astype(int))
                if len(lablen) > 2:
                    srt = np.sort(lablen)[::-1].astype(float)
                    ratio = srt[:-1] / srt[1:]
                    thres = np.argmax(ratio) + 1
                    objects = np.argsort(lablen)[::-1][:thres]
                    #objects = np.argsort(lablen)[-n_clusters:]
                    obj_len = lablen[objects]

                #labs = np.array(labs)
                #new_labs = np.zeros(len(labs))
                #for objno, obj in enumerate(objects):
                    #if len(simils[obj]) < 2:
                        #stds.append(0)
                    #else:
                        #stds.append(np.std(simils[obj]))

                    #new_labs[labels == obj] = objno+1

                #for sm in simils:
                    #if len(sm) < 2:
                        #stds.append(0)
                    #else:
                        #stds.append(np.std(sm))
                #mean_std = np.mean([np.std(sm) ])


                #stds = []
                #for lab in np.unique(labs).astype(int):
                    ##print lab
                    #sims = sim[np.array(lab)]
                    #sel = sim[sims>0]
                    ##import pdb; pdb.set_trace()
                    #if len(sel) < 2:
                        #std = 0
                    #else:
                        #std = np.std(sel)
                    #stds.append(std)
                ##import pdb; pdb.set_trace()

                #stds = np.array(stds)
                #sel = stds[stds>0]
                #if len(sel) == 0:
                    #mean_std = np.inf
                #else:
                    #mean_std = np.mean(sel)
                #mean_stds.append(mean_std)
                #print edges, len(np.unique(labs)), mean_std
                    print edges, len(np.unique(labs)), obj_len

                    #import pdb; pdb.set_trace()
                    #import pdb; pdb.set_trace()
                    labels.append(labs)

                    if len(obj_len) > n_clusters:
                        #if last_obj_len == n_clusters:
                        #if len(obj_len) != n_clusters:
                        try:
                            labs = last_labs
                        except:
                            pass
                        break

                    last_obj_len = len(obj_len)
                    last_labs = labs

            #import pdb; pdb.set_trace()
            #mean_stds = np.array(mean_stds)
            #ind = np.argmin(mean_stds)
            #print ind, bin_edges[::-1][ind]
            #labels = labels[ind]
        else:
            # snakes: 0.00112665583835 or 0.0011439890051
            # textures: .52
            labs, simils = self._simcluster(stim, sim, simthres=simthres)

        labels = labs
        lablen = np.bincount(labels.astype(int))
        objects = np.argsort(lablen)[-n_clusters:]
        labels = np.array(labels)
        new_labs = np.zeros(len(labels))
        #import pdb; pdb.set_trace()
        for objno, obj in enumerate(objects):
            new_labs[labels == obj] = objno+1
        #new_labs = labels

        #import pdb; pdb.set_trace()
        #for lab in np.unique(labels):
            #labels[labels==lab]
        #ind = 0
        #while len(labels) > n_clusters or start:
            #print radius
        #for radius in range(1,21):
            #print radius
            #lab = self._affcluster(stim, radius=radius)
            #labels.append(lab)
            #lablen.append(len(np.unique(lab)))
        #print 'cluster sizes:', lablen
        #ind = np.argmax(np.array(lablen[:-1]) / np.array(lablen[1:])) + 1
        #print 'selected cluster:', ind

        #labels = np.mean(labels, axis=0).astype(int)
        #import pdb; pdb.set_trace()

        #self.plot_cluster(stim, new_labs, ax=ax)
        return new_labs

    def _simcluster(self, stim, sim, simthres=.3):
        #inds = stim.indices
        X = sim.copy()
        simils = []
        simil = []


        use_inds = range(len(X))
        np.random.shuffle(use_inds)
        i = use_inds.pop()
        X[:,i] = -np.inf
        clusters = []
        cluster = [i]

        while len(use_inds) > 0:

            sel = X[np.array(cluster)]
            mx = sel>simthres
            #import pdb; pdb.set_trace()
            if np.sum(mx) > 0:
                inds = np.nonzero(np.any(mx, 0))[0]
                simil.extend(sel[mx].tolist())
                #import pdb; pdb.set_trace()
                #if 564 in inds: import pdb; pdb.set_trace()
                cluster += inds.tolist()
                #j = np.argmax(sel)
                #j = np.unravel_index(j, sel.shape)[-1]
                for ind in inds:
                    del use_inds[use_inds.index(ind)]
                X[:,inds] = -np.inf
                #cluster.append(j)
                #i = j
            else:
                clusters.append(cluster)
                simils.append(simil)
                simil = []
                np.random.shuffle(use_inds)
                i = use_inds.pop()
                X[:,i] = -np.inf
                cluster = [i]

        labels = np.zeros(len(X))
        #import pdb; pdb.set_trace()
        for k, cluster in enumerate(clusters):
            #if len(cluster) > 100:
                labels[np.array(cluster)] = k
            #else:
                #labels[np.array(cluster)] = -1
        return labels, simils

    def plot_cluster(self, stim, labels, ax=None):
        if ax is None:
            ax = plt.subplot(111)

        colors = self.get_colors(len(np.unique(labels)))
        colors[0] = (0,0,0,1)  # black
        for i, label in enumerate(np.unique(labels)):
            try:
                ax.plot(stim.indices[labels==label,1],
                    stim.indices[labels==label,0],
                    'o', color=colors[i])
            except:
                import pdb; pdb.set_trace()
        ax.set_xlim((0, stim.shape[1]-1))
        ax.set_ylim((stim.shape[0]-1, 0))

    def get_colors(self, n):
        thisCmap = mpl.cm.get_cmap('jet')
        norm = mpl.colors.Normalize(0, 1)
        z = np.linspace(0, 1, n+2)
        z = z[1:-1]
        colors = thisCmap(norm(z))
        #np.random.shuffle(colors)
        return colors

    def get_clusters(self, Z, thres=0, n_clusters=None):
        if n_clusters is not None:
            thres = Z[-n_clusters, 2]

        g = [[j] for j in range(len(Z)+1)]
        i = 0
        while Z[i,2] <= thres:
            x, y = Z[i,:2].astype(int)
            g.append( g[x] + g[y] )
            g[x] = []
            g[y] = []
            i += 1
        clusters = [gg for gg in g if len(gg)>0 ]
        return clusters

    def _thres_im(self, image, thres=0):
        return np.where(image<thres, np.zeros(image.shape), image)

    def sim2im(self, stim, sim, pos):
        image = np.zeros(stim.shape)
        for noi, p in enumerate(stim.indices):
            if p[0] == pos[0] and p[1] == pos[1]:
                break
        for noj, p in enumerate(stim.indices):
            image[p[0], p[1]] = sim[noi, noj]

        plt.imshow(image)
        plt.plot(pos[1], pos[0], 'o', c='g')
        plt.colorbar()
        plt.show()

    def model(self, stim, nelems=1000, load=True, n_clusters=3):
        """Steps of processing
        """
        # Step 1: extract edges (contrast and orientations)
        stim_read = False
        fname = stim.name+'.pkl'
        if load:
            try:
                stim = pickle.load(open(fname,'rb'))
            except:
                pass
            else:
                stim_read = True
                print 'stim loaded from %s ' % fname
        if not stim_read:
            self.compute_edges(stim, stim.image)
            pickle.dump(stim, open(fname,'wb'))

        # Step 2: select locally maximally responding units
        resps = []
        resp = stim.contrasts
        for jj in range(3):
            stim, grouping, resp2 = self._model(stim, resp, nelems=nelems, n_clusters=n_clusters, niter=3)
        #weights = resp / np.sum(resp)
            #ch = self.weighted_choice(resp, 3000)
            #plt.imshow(resp, cmap='gray');plt.colorbar();
            ##plt.scatter(ch[:,1], ch[:,0])
            #plt.show()
        #import pdb; pdb.set_trace()
            resps.append(resp2)

        resp = np.max(np.array(resps), 0)
        #plt.imshow(resp, cmap='gray');plt.show()
        #import pdb; pdb.set_trace()
        stim, grouping, resp = self._model(stim, resp, nelems=np.max((1000,.1*nelems)),
                                n_clusters=n_clusters, niter=1, method='thres')
        return stim, grouping, resp

    def weighted_choice(self, weights, n=1):
        """Return a random item from objects, with the weighting defined by weights
        (which must sum to 1).

        From http://stackoverflow.com/a/10803136 and
        http://code.activestate.com/recipes/498233-weighted-choice-short-and-easy-numpy-version/
        """
        weights = weights / np.sum(weights)
        cs = np.cumsum(weights.ravel()) #An array of the weights, cumulatively summed.
        rnum = np.random.random(n) #return array of uniform numbers (from 0 to 1) of shape sh
        inds = np.searchsorted(cs, rnum)
        #inds = [np.sum(cs < np.random.rand()) for i in range(n)] #Find the index of the first weight over a random value.
        #plt.imshow(weights, cmap='gray'); plt.colorbar(); plt.show()
        #import pdb; pdb.set_trace()
        inds = np.unique(inds)
        inds = np.unravel_index(inds, weights.shape)
        inds = np.vstack(inds).T

        return inds


    def _model(self, stim, resp, nelems=1000, n_clusters=3, niter=3, method='weighted'):
        for ii in range(niter):
            #if ii == 0:
            if method=='weighted':
                stim.indices = self.weighted_choice(resp, n=nelems)
                print stim.indices.shape
            else:
                stim.indices = self.select_inds(resp, nelems=nelems, sel='thres')
                #plt.imshow(resp, cmap='gray');plt.show()
            #else:
                #resp = stim.good_cont*stim.proximity*stim.contrasts
                #stim.indices = self.select_inds(resp, nelems=2000,
                                #sel='random', size=3)
                #resp = stim.good_cont#*stim.proximity*stim.similarity_contrasts


            #import pdb; pdb.set_trace()
            # Step 3: compute grouping strengths between the extracted units
            #for aff in ['proximity', 'similarity_ori', 'good_cont']:
            stim.contrasts2, stim.contrasts_group = self.compute_grouping(stim, affinity=None)
            stim.proximity, stim.proximity_group = self.compute_grouping(stim, affinity='proximity')
            stim.similarity_contrasts, stim.similarity_contrasts_group = self.compute_grouping(stim, affinity='similarity_contrasts')
            stim.similarity_oris, stim.similarity_oris_group = self.compute_grouping(stim, affinity='similarity_ori')
            stim.good_cont, stim.good_cont_group = self.compute_grouping(stim, affinity='good_cont')

            if stim.name == 'dotlats':
                grouping = stim.contrasts_group * stim.good_cont_group
            elif stim.name == 'snakes':
                grouping = stim.contrasts_group * stim.good_cont_group
            elif stim.name == 'textures':
                grouping = stim.contrasts_group * stim.similarity_contrasts_group
                       #stim.proximity_group
                       #stim.similarity_contrasts_group *\
                       #stim.similarity_oris_group
                       # * \
            #if ii != niter-1:
                #nbins = 5000
                #hist, bin_edges = np.histogram(grouping[grouping>0].ravel(),bins=nbins)
                #last = np.nonzero(np.cumsum(hist[::-1]) > 1000)[0][0]
                #threshold = bin_edges[nbins-last-1]
                #inds = np.where(grouping>threshold)
                #inds = np.unique(inds[0])
                ##import pdb; pdb.set_trace()
                #stim.indices = stim.indices[inds]
                #print "final inds:", stim.indices.shape
                ##labels = self.affcluster(stim, grouping, n_clusters=n_clusters)
                ##stim.indices = stim.indices[np.array(labels)>0]

            #grouping = stim.proximity_group * stim.similarity_contrasts_group *\
                         #stim.similarity_oris_group #* stim.good_cont_group
        #import pdb; pdb.set_trace()
        #plt.imshow(np.max(np.array(af),0), cmap='gray'); plt.colorbar(); plt.show()
        #stim.oriaf, stim.origroup = self.similarity(stim, stim.contrasts,
                                                     #affinity=self.ori_field)
        #stim.af = stim.oriaf
        #stim.group = stim.origroup
        #stim.af, stim.group = self.similarity(stim, stim.contrasts,
                                                  #affinity=self.ernst_full)
        #nelems2 = np.min((2000,2*nelems))
        #mx, stim.indices = self.sparse_local_max(stim.af, size=3,
                            #nelems=nelems2)#, nelems=1500)

        #stim.af, stim.group = self.similarity(stim, stim.af)
        #import pdb; pdb.set_trace()

        #meangr = np.mean(grouping, 1)
        #im = np.zeros(stim.shape)
        ##im[stim.indices[:,0],stim.indices[:,1]] = meangr
        ##plt.imshow(im, cmap='gray'); plt.colorbar();
        #plt.imshow(stim.image, cmap='gray')
        #plt.scatter(stim.indices[:,1],stim.indices[:,0],
            #s=meangr*10000,c=meangr)
        #plt.show()
        #import pdb; pdb.set_trace()
            if stim.name == 'dotlats':
                resp = stim.contrasts*stim.good_cont
            elif stim.name == 'snakes':
                resp = stim.contrasts*stim.good_cont
            elif stim.name == 'textures':
                resp = stim.contrasts*stim.similarity_oris
                   #stim.similarity_contrasts * stim.similarity_oris
            #plt.imshow(resp, cmap='gray');plt.colorbar();
            #plt.scatter(stim.indices[:,1], stim.indices[:,0])
            #plt.show()
            #import pdb; pdb.set_trace()
        return stim, grouping, resp


    def run(self, stim='snakes', task=None, n_clusters=3, density=.01):
        """Get stimulus, process, and plot the output
        """
        stim = Image(stim)
        nelems = int(stim.image.size * density)
        self.deg2px = stim.deg2px
        stim, grouping, resp = self.model(stim, nelems=nelems, n_clusters=n_clusters)
        #self.affcluster(stim, grouping, n_clusters=n_clusters)
        self.plot_results(stim, grouping, resp, n_clusters=n_clusters)

    def plot_results_old(self, stim, grouping, n_clusters=3):
        """Plots results of for each stimulus attribute
        """
        fig = plt.figure()
        axes = ImageGrid(fig, 111, nrows_ncols=(4,3), share_all=True,
                         cbar_mode="each", axes_pad=.5, cbar_pad=0)

        ### Input image ###
        axes[1].imshow(stim.image, cmap='gray')
        axes[1].set_title('original image')

        axes[2].imshow(stim.image, cmap='gray')
        axes[2].set_title('original image, clustering')
        #_, stim_sim = self.similarity(stim, stim.image)
        #self.cluster(stim, stim_sim, axes[2], n_clusters=n_clusters)

        ### Orientations ###
        gabors = self.plot_gabor_oris()
        axes[3].imshow(gabors, cmap='gray')
        axes[3].set_title('gabor orientations')

        self.plot_arrows(stim, stim.contrasts, stim.oris, axes[4])

        axes[5].imshow(stim.image, cmap='gray')
        self.plot_arrows(stim, stim.contrasts, stim.oris, axes[5])
        axes[5].set_title('orientations')

        ### Gabors ###
        gabors = self.plot_gabor_sizes(stim)
        axes[6].imshow(gabors, cmap='gray')
        axes[6].set_title('gabor sizes')

        axes[7].imshow(stim.contrasts, cmap='gray')
        axes[7].set_title('filtered with gabors')

        axes[8].imshow(stim.contrasts, cmap='gray')
        axes[8].set_title('clustering')
        #ff, contrasts_sim = self.similarity(stim, stim.contrasts)
        #self.affcluster(stim, contrasts_sim, axes[8], n_clusters=n_clusters)
        #import pdb; pdb.set_trace()

        ### Association field ###
        gc = self.plot_assoc_field(stim)
        im = axes[9].imshow(gc, cmap='gray')
        axes[9].set_title('good continuation')
        axes[9].cax.colorbar(im)

        #thresaf = self.thres_im(stim.af, thres=.0012)*100
        im = axes[10].imshow(stim.good_cont, cmap='gray')
        axes[10].set_title('good continuation')
        axes[10].cax.colorbar(im)

        im = axes[11].imshow(stim.good_cont, cmap='gray')
        axes[11].set_title('clustering')
        #import pdb; pdb.set_trace()
        self.affcluster(stim, grouping, axes[11], n_clusters=n_clusters)
        #allinds = range(len(stim.indices))
        #n = 1000
        #allp = np.zeros(n)
        #allsel = np.zeros((n,40))
        #for i in range(n):
            #np.random.shuffle(allinds)
            ##import pdb; pdb.set_trace()
            #inds = allinds[:40]
            #allp[i] = np.sum(stim.group[inds,inds])
            #allsel[i] = inds
        #sel = np.argmax(allp)
        #thisinds = stim.indices[allsel[sel]]
        #plt.plot(thisinds[0], thisinds[1], 'o')



        ### Figure ground ###
        #im = axes[13].imshow(stim.fg, cmap='gray')
        #axes[13].set_title('figure-ground')
        #axes[13].cax.colorbar(im)

        #im = axes[14].imshow(stim.fg, cmap='gray')
        #axes[14].set_title('clustering')
        #self.cluster(stim, stim.fggroup, axes[11], n_clusters=n_clusters)

        plt.show()

    def plot_results(self, stim, grouping, resp, n_clusters=3):
        """Plots results of for each stimulus attribute
        """
        fig = plt.figure()
        axes = ImageGrid(fig, 111, nrows_ncols=(5,3), share_all=True,
                         cbar_mode="each", axes_pad=.5, cbar_pad=0)

        ### Input image ###
        axes[0].imshow(stim.image, cmap='gray')
        axes[0].set_title('original image')

        axes[1].set_title('total grouping probabilities')
        meangr = np.mean(grouping, 1)
        axes[1].imshow(stim.image, cmap='gray')
        axes[1].scatter(stim.indices[:,1],stim.indices[:,0],
            s=meangr*10000,c=meangr)

        im = axes[2].imshow(stim.image, cmap='gray')
        axes[2].set_title('clustering')
        labels = self.affcluster(stim, grouping, axes[2], n_clusters=n_clusters)
        self.plot_cluster(stim, labels, ax=axes[2])

        ### Proximity ###
        #gabors = self.plot_gabor_sizes(stim)
        #axes[6].imshow(gabors, cmap='gray')
        #axes[6].set_title('gabor sizes')

        #axes[3].imshow(stim.contrasts, cmap='gray')
        #axes[3].set_title('proximity')

        axes[5].imshow(stim.proximity, cmap='gray')
        axes[5].set_title('proximity')
        axes[5].cax.colorbar(im)

        ### Orientations ###
        gabors = self.plot_gabor_oris()
        axes[6].imshow(gabors, cmap='gray')
        axes[6].set_title('gabor orientations')

        axes[7].imshow(stim.image, cmap='gray')
        self.plot_arrows(stim, stim.contrasts, stim.oris, axes[7])
        axes[7].set_title('orientations')

        im = axes[8].imshow(stim.similarity_oris, cmap='gray')
        axes[8].set_title('ori similarity')
        axes[8].cax.colorbar(im)

        ### Gabors ###
        gabors = self.plot_gabor_sizes(stim)
        axes[9].imshow(gabors, cmap='gray')
        axes[9].set_title('gabor sizes')

        axes[10].imshow(stim.contrasts, cmap='gray')
        axes[10].set_title('contrasts')

        im = axes[11].imshow(stim.similarity_contrasts, cmap='gray')
        axes[11].set_title('contrast similarity')
        axes[11].cax.colorbar(im)

        ### Association field ###
        gc = self.plot_assoc_field(stim)
        im = axes[12].imshow(gc, cmap='gray')
        axes[12].set_title('good continuation')
        axes[12].cax.colorbar(im)

        #thresaf = self.thres_im(stim.af, thres=.0012)*100
        im = axes[13].imshow(stim.good_cont, cmap='gray')
        axes[13].set_title('good continuation')
        axes[13].cax.colorbar(im)

        im = axes[14].imshow(resp, cmap='gray')
        axes[14].set_title('final grouping')
        axes[14].cax.colorbar(im)


        plt.show()


    def report(self, stim, n_clusters=2, density=.01):
        stimuli = [
                    ('dotlats',7),
                    ('snakes',2),
                    ('textures',3)
                    ]
        fig = plt.figure()
        axes = ImageGrid(fig, 111, nrows_ncols=(len(stimuli),2), share_all=True,
                         cbar_mode=None, axes_pad=.5, cbar_pad=0)

        for i, (stim, n_clusters) in enumerate(stimuli):
            stim = Image(stim)
            nelems = int(stim.image.size * density)
            self.deg2px = stim.deg2px
            stim, grouping, resp = self.model(stim, nelems=nelems, n_clusters=n_clusters)

            ### Plot ###
            im = axes[3*i].imshow(stim.image, cmap='gray')
            axes[3*i].set_title('input image')
            #axes[3*i].cax.colorbar(im)

            #im = axes[3*i+1].imshow(grouping, cmap='gray')
            #axes[3*i+1].set_title('probability map')
            #axes[3*i+1].cax.colorbar(im)

            im = axes[3*i+1].imshow(stim.image, cmap='gray')
            axes[3*i+1].set_title('clustering')
            labels = self.affcluster(stim, grouping, axes[3*i+1], n_clusters=n_clusters)
            self.plot_cluster(stim, labels, ax=axes[3*i+1])
        plt.show()


class Image(object):

    def __init__(self, stim=None, nelems=1000):
        super(Image, self).__init__()
        self.name = stim
        if stim is None:
            self.image = self.get_image()
        else:
            self.image = self.get_image(which=stim)
        self.shape = self.image.shape
        self.deg2px = 51. # how many pixels is 1 deg; I made it up, of course
        self.size_deg = self.shape[0] / self.deg2px

    def set_features(self):
        # features
        self.ori = self.get_ori()
        self.contrast = self.stim
        self.color = None  # not implemented yet

        # other
        self.indices = None


    def get_image(self, which='snake1'):
        stimdict = {
            'dotlat1': '005', 'dotlat2': '006b', 'dotlats': ['005','006b'],
            'snake1': '010', 'snake2': '011', 'snakes': ['010', '011'],
            'texture1': '030', 'texture2': '031', 'textures': ['030','031'],
            'contour1': '035a', 'contour2': '035b', 'contours': ['035a', '035b'],
            'frcont1': '035', 'frcont2': '036', 'frconts': ['035', '036']
            }
        if which not in stimdict:
            ims = which
        else:
            ims = stimdict[which]
        if isinstance(ims, str):
            ims = [ims]

        stims = []
        for im in ims:
            if which not in stimdict:
                try:
                    stim = scipy.misc.imread('visgest101/%s' % im, flatten=True)
                except:  # guess the extension
                    stim = scipy.misc.imread('visgest101/%s.png' % im, flatten=True)
            else:
                stim = scipy.misc.imread('visgest101/%s.png' % im, flatten=True)
            stim = scipy.misc.imresize(stim, (256, 256))
            stims.append(np.asarray(stim)*1./255)
        stim = np.hstack(stims)
        #import scipy.misc
        #stim = scipy.misc.lena()
        #stim = np.hstack((stim, 255*np.ones((stim.shape[0], 50))))
        if which == 'dotlats':
            stim = np.hstack([stim[75:180,75:180],
                        stim[75:180,256+75:256+180]])
        elif which == 'contours':
            #import pdb; pdb.set_trace()
            stim2 = np.ones(stim.shape)
            stim = scipy.misc.imresize(stim, .8) * 1/255.
            d = ((stim2.shape[0] - stim.shape[0])/2,
                 (stim2.shape[1] - stim.shape[1])/2)
            stim2[d[0]:stim.shape[0]+d[0], d[1]:stim.shape[1]+d[1]] = stim
            stim = stim2
            #import pdb; pdb.set_trace()
        return stim

    def radialfreq(self, r_mean=40, A=0.2, om=5, phi=0):
        """
        Generates a radial frequency pattern.

        .. :warning: not tested
        """
        theta = np.arange(100)/100. * 2 * np.pi
        r = r_mean * (1 + A * np.sin(om * theta + phi))
        r = (r + np.roll(r, -1)) / 2
        r1 = r
        r2 = np.roll(r, -1)
        theta1 = theta
        theta2 = np.roll(theta, -1)
        tangent = np.arctan2(r1*np.sin(theta1) - r2*np.sin(theta2), r1*np.cos(theta1) - r2*np.cos(theta2))
        r = (r1 + r2) / 2
        theta = (theta1 + theta2) / 2

        rf = np.zeros((2, r_mean*4,r_mean*4))
        x = rf.shape[1]/2 + (r*np.cos(theta)).astype(int)
        y = rf.shape[1]/2 - (r*np.sin(theta)).astype(int)  # have to invert
        rf[0][y,x] = 1
        rf[1][y,x] = tangent % np.pi #[y,x]
        indices = np.vstack((y,x)).T
        return rf, indices


if __name__ == '__main__':
    gmin = GMin()
    gmin.report('textures', n_clusters=3, density=.03)
