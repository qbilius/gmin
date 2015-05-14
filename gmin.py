"""
gmin: An open, minimalist mid-level framework

This is a simplistic prototype implementation of the similarity and pooling
framework proposed in

  Kubilius, J., Wagemans, J., Op de Beeck, H. P. (2014). A conceptual framework 
  of mid-level vision. *Frontiers in Computational Neuroscience*, 8, 158. 
  doi: `10.3389/fncom.2014.00158 <http://doi.org/10.3389/fncom.2014.00158>`_

Author: Jonas Kubilius (qbilius@gmail.com)
License: GNU GPL 3+
"""

import urllib, copy, StringIO

import numpy as np
import scipy.misc, scipy.ndimage
from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

rc_params = {'image.interpolation': 'none'}
plt.rcParams.update(rc_params)

def layer1(im, box=12):
    """
    First layer where the intial feature detection is performed
    
    :Args:
        im (numpy array)
            input image
    :Kwargs:
        box (int, default: 12)
            The image is divided into a grid of box x box pixel patches, and all
            features are computed in that patch.
    :Returns:
        - edges (numpy array)
            A zero-padded array with dominant orientations
        - clusters1 (numpy array)
            The initial division of the image into clusters. Each patch is a 
            unique cluster for now.
        - fms1 (dict)
            Feature map where each cluster id has a list of features.
        - crop
            A tuple of coordinates to crop the padding
    """

    ## edge detection
    mags_scsp, absoris_scsp = compute_edges(im)
    edges = pool_edges(im, mags_scsp, absoris_scsp, box=box)
    edges[:,:,2] %= np.pi  # transform absori to ori

    ## similarity and pooling
    edges, crop = pad(edges, (9,9,edges.shape[-1]))
    # initial cluster assignment
    clusters1 = np.arange(edges.shape[0]*edges.shape[1]).astype(int) + 1
    clusters1 = clusters1.reshape((edges.shape[0],edges.shape[1]))
    fms1 = dict([(c, np.squeeze(edges[clusters1==c])) for c in np.unique(clusters1)])

    return edges, clusters1, fms1, (crop[0], crop[1])

def layer2(clusters1, fms1, ts):
    """
    Second and higher layers
    
    :Args:
        - clusters1 (numpy array)
            Clusters from the previous layer
        - fms 1 (dict)
            Feature maps from the previous layer
        - ts
            A list of dict containing thresholds for various features
    :Returns:
        - clusters
        - fms
    """
    cluster_list = []
    clusters = clusters1.copy()
    fms = copy.copy(fms1)

    i = 0
    for box, t in zip([3,9,None], ts):
        clusters, fms, sim = stride2dzip(clusters, fms, box=box, func=pooling1,
                                         kwargs={'thres': t})

    cluster_list.append((clusters, ts))

    return clusters, fms

def compute_edges(im, scales=[5, 11, 17, 23]):
    """
    Multiscale edge detection
    
    :Args:
        im (numpy.array)
    :Kwargs:
        scales (list, default: [5, 11, 17, 23])
    :Returns:
        - mags_scsp (3D numpy.array with scale in the first dimension)
            Response magnitudes at all scales
        - absoris_scsp (3D numpy.array with scale in the first dimension)
            Absolute orientation (between 0 and 2*pi) at all scales
    """
    scalespace_hor = np.zeros((len(scales), im.shape[0], im.shape[1]))
    scalespace_ver = np.zeros_like(scalespace_hor)

    # extract edges using Gabor filters
    for sno, scale in enumerate(scales):
        g = gabor(size=scale, theta=0, phase=np.pi/2)
        scalespace_hor[sno] = scipy.ndimage.correlate(im, g) / scale
        g = gabor(size=scale, theta=np.pi/2, phase=np.pi/2)
        scalespace_ver[sno] = scipy.ndimage.correlate(im, g) / scale

    mags_scsp = np.sqrt(scalespace_hor**2 + scalespace_ver**2)
    absoris_scsp = np.arctan2(scalespace_ver, scalespace_hor)
    absoris_scsp = (absoris_scsp + np.pi) % (2*np.pi)  # put in the range [0,2*pi]

    return mags_scsp, absoris_scsp

def gabor(size=42, theta=0, gamma=1, phase=0):
    """
    Returns a Gabor filter as in Mutch and Lowe, 2006.

    :Kwargs:
        - size (int or float, default:42)
            Size of the Gabor patch (equal to 1/sf)
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
    sigma = size/6.
    r = int(sigma*3)  # there is nothing interesting 3 STDs away

    i,j = np.meshgrid(np.arange(-r,r+1),np.arange(r,-r-1,-1), indexing='ij')
    # rotate coordinates
    ii = i*np.cos(theta) - j*np.sin(theta)
    jj = i*np.sin(theta) + j*np.cos(theta)
    g = np.exp( -(ii**2 + (gamma*jj)**2) / (2*sigma**2) ) * np.cos( 2*np.pi*ii/r + phase)

    g -= np.mean(g)  # mean 0
    g /= scipy.linalg.norm(g) #np.sum(g**2) # energy 1
    # g = g.T
    # g /= np.max(np.abs(g))
    # g[np.abs(g)<.001] = 0  # remove spillover
    # g -= np.min(g)  # all positive

    return g

def pad(m, box):
    """
    Pad an array with zeros.
    
    The array will be padded with zeros such that the resulting array can be 
    divided perfectly into patches of size *box*.
    
    :Args:
        - m (numpy.array)
            An array that needs to be zero-padded.
        - box (int or list)
            If an int, all dimensions will be padded using the same box size.
    :Returns:
        - pm (numpy.array)
            Zero-padded array
        - s (a tuple of numpy slices)
            Slices for cropping the padded array back to the original shape.
    """
    try:
        iter(box)
    except:
        box = np.repeat(box, m.ndim)

    nextra = [sh % b for sh, b in zip(m.shape, box)]
    pm = np.zeros([sh/b*b + b*min(n,2) for sh, b, n in zip(m.shape, box, nextra)])
    pm = pm.astype(m.dtype)
    s = tuple([np.s_[(psh-sh)/2: (psh+sh)/2] for sh, psh in zip(m.shape, pm.shape)])
    pm[s] = m
    return pm, s

def stride2dzip(m, f, box=24, func=np.max, ax=None, args=(), kwargs={}):
    """
    """
    
    if box is None:
        box = (m.shape[0], m.shape[1], 2)
    try:
        iter(box)
    except:
        box = (box, box) + m.shape[2:]

    pm = m
    pf = f

    clusters = np.zeros_like(pm)
    fms = {}
    sims = []
    ccount = 0
    for mi in np.arange(0, pm.shape[0], box[0]):
        for mj in np.arange(0, pm.shape[1], box[1]):
            s = np.s_[mi:mi+box[0], mj:mj+box[1]]
            clusters[s], fms_out, sim = func(pm, pf, s, ccount, *args, **kwargs)
            fms.update(fms_out)
            sims.append(sim)
            ccount = np.max(clusters)
            if ax is not None:
                clusterplot(clusters[s], s, ax=ax)

    return clusters, fms, sims

def stride2d(m, box=24, func=np.max, args=(), kwargs={}):
    try:
        iter(box)
    except:
        box = (box, box) + m.shape[2:]

    pm, _ = pad(m, box)
    out = []
    for mi in np.arange(0, pm.shape[0], box[0]):
        outi = []
        for mj in np.arange(0, pm.shape[1], box[1]):
            s = np.s_[mi:mi+box[0], mj:mj+box[1]]
            outi.append(func(pm, s, *args, **kwargs))
        out.append(outi)

    try:
        out = np.array(out)
    except:
        pass

    return out

def _ori_dist(a,b):
    d = np.abs(a-b) % np.pi
    d[d>np.pi/2] = np.pi - d[d>np.pi/2]
    d /= np.pi/2
    return d

def _lum_sim(feats):
    sim = 1 - pairwise_distances(np.reshape(feats, (-1,1)))
    return sim

def _ori_sim(feats):
    dist = pairwise_distances(np.reshape(feats, (-1,1)), metric=_ori_dist)
    sim = 1-dist
    return sim

def _mag_sim(feats):
    sim = 1 - pairwise_distances(np.reshape(feats, (-1,1)))
    return sim

def _mag(feats):
    """
    Are both magnitudes large enough?
    """
    sim = pairwise_distances(np.reshape(feats, (-1,1)), metric=lambda a,b: a*b)
    return sim

def _pool1(features):
    return np.median(features, axis=0)

def pooling1(clusters, fms, s, ccount=0, thres=.9, ax=None):
    cls = clusters[s]
    unq = np.unique(cls)
    feats = np.array([fms[c] for c in unq])
    sim_lum = _lum_sim(feats[:,0])
    sim_mag = _mag_sim(feats[:,1])
    sim_ori = _ori_sim(feats[:,2])
    magmtx = _mag(feats[:,1])
    sim = np.dstack([sim_lum, sim_mag, sim_ori, magmtx])
    cluster_inds, _ = simcluster(sim, simthres=thres)

    clusters2 = np.zeros_like(cls)
    fms2 = {}
    for newc, cluster2 in enumerate(cluster_inds):
        sel = np.array([c for i,c in enumerate(unq) if i in cluster2])
        tmp = []
        newcc = newc + ccount + 1  # zero is reserved for padding
        for c in sel:
            clusters2[cls==c] = newcc
            tmp.append(fms[c])
        fms2[newcc] = _pool1(tmp)

    return clusters2, fms2, sim

def pool_edges(lum, mags_scsp, absoris_scsp, box=24):
    boxp = (mags_scsp.shape[0], box, box)
    mags_scsp_p, _ = pad(mags_scsp, boxp)
    absoris_scsp_p, _ = pad(absoris_scsp, boxp)

    return stride2d(lum, box=box, func=_pool_edges,
                    args=(mags_scsp_p, absoris_scsp_p))

def _pool_edges(lum, s, mags_scsp, absoris_scsp):

    # get the mean luminance in the patch
    lump = np.mean(lum[s])

    # get the strongest edge response and it's position
    mags = np.sum(mags_scsp, axis=0)
    sel = mags[s]
    pos = np.argmax(sel)
    pos = np.unravel_index(pos, sel.shape)
    mag = sel[pos]

    # get the absori of that maximal edge
    pos_sc = np.argmax(mags_scsp[:, s[0], s[1]][:, pos[0], pos[1]])
    absori = absoris_scsp[pos_sc, s[0], s[1]][pos]
    return lump, mag, absori

def simcluster(sim, simthres={}):
    """
    Clustering based on feature similarity.
    
    Clusters features according to their similarity. In this implementation,
    we first check if orientation magnitudes are both high and if so, we group
    the two input locations if their orientations are similar. If magnitudes are
    below the threshold, we instead check if their luminances are similar.
    The current implementation only uses the following features:
        - magmtx - magnitude strength (orientation magnitude of one location 
          times orientation magnitude of another)
        - ori - orientation similarity
        - lum - luminance similarity
        
    :Args:
        - sim (numpy.array)
            A square similarity matrix. Similarities of each feature are 
            contained in the third dimension.
    :Kwargs:
        - simthres (dict)
            Similarity thresholds
        
    """
    X = sim.copy()
    use_inds = range(len(X))
    simils = []
    clusters = []

    while len(use_inds) > 0:
        use_inds = np.array(use_inds)
        order = np.argsort(np.diag(X[:,:,-1])[use_inds])
        use_inds = use_inds[order].tolist()
        i = use_inds.pop()  # start from a particular row
        simil = [X[i,i]]
        cluster = [i]
        X[:,i] = -np.inf  # get rid of symmetric values in the column

        sel = X[np.array(cluster)]  # get particular rows
        if np.any(sel[:,:,3] > simthres['magmtx']):  # mags are large enough
            mx = np.logical_and(sel[:,:,3] > simthres['magmtx'],
                                sel[:,:,2] > simthres['ori'])  # oris are similar
        else:
            mx = sel[:,:,0] > simthres['lum']

        inds = np.nonzero(np.any(mx, axis=0))[0]  # get indices of similar
        cluster += inds.tolist()
        simil += sel[mx].tolist()
        clusters.append(cluster)
        simils.append(simil)

        for ind in inds:
            del use_inds[use_inds.index(ind)]
        X[:,inds] = -np.inf  # get rid of those columns

    labels = np.zeros(len(X)).astype(int)

    for k, cluster in enumerate(clusters):
        labels[np.array(cluster)] = k

    return clusters, (labels, simils)


def plot_clusters(im, clusters):
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.imshow(im, cmap='gray')
    ax = fig.add_subplot(122)
    ax.imshow(clusters)
    return fig

def fig2str(fig):
    """
    Converts a Matplotlib figure to a string.
    
    Useful for web when you don't want to save and serve the figure.
    """
    canvas = FigureCanvasAgg(fig)
    png_output = StringIO.StringIO()
    canvas.print_png(png_output)
    png_output = png_output.getvalue().encode("base64")
    encoded = urllib.quote(png_output.rstrip('\n'))
    return encoded

def run(f, show=True):
    """
    Main function to call.
    
    :Args:
        f (file name or file object)
            Path to an a input image or a file object containing the image.
    :Kwargs:
        show (bool, default: True)
            If True, shows the input image and the corresponding output. 
            Othwerwise returns this figure encoded as a string.
        
    """
    # thresholds
    ts = [{'lum': .9, 'mag': None, 'ori': .85, 'magmtx': .04},
          {'lum': .9, 'mag': None, 'ori': .85, 'magmtx': .04},
          {'lum': .9, 'mag': None, 'ori': .85, 'magmtx': .04}]

    # prepare the image
    im = scipy.misc.imread(f, flatten=True)
    scale = 256./min(im.shape)
    im = scipy.misc.imresize(im, [int(i*scale) for i in im.shape])
    im = np.asarray(im)*1./255

    # time for gmin
    edges, clusters, fms, crop = layer1(im)
    clusters, fms = layer2(clusters, fms, ts=ts)
    clusters = clusters[crop][1:-1, 1:-1]
    
    fig = plot_clusters(im, clusters)
    if show:
        plt.show()
    else:
        return fig2str(fig)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    args = parser.parse_args()
    run(args.image, show=True)
