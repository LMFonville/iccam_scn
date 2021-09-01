import os
import glob

import numpy as np
import networkx as nx
import scona as scn
import pandas as pd


def checkPermutations(file):
    # read in file that contains each shuffle of group membership and check for unique combinations
    perms = pd.read_csv(file, header=None, sep = ' ')
    perms_u = perms.drop_duplicates()
    print("There were %i (out of %i) unique permutations" % (len(perms_u), len(perms)))

def getPermutationsFast(inputpath, overwrite=False):
    # collate outputs from each permutation in a single file
    # faster implementation using list comprehension
    global_permutations = glob.glob(inputpath + '/global/*csv')
    global_permutations = sorted(global_permutations)
    local_permutations = glob.glob(inputpath + '/local/*csv')
    local_permutations = sorted(local_permutations)
    # either prepare new dataframe or set getlocal and getglobal to false to avoid loading data
    if not os.path.isfile(inputpath + '/global_permutations.csv') or overwrite:
        getglobal=True
    else:
        getglobal=False
        global_perms = pd.read_csv(inputpath + '/global_permutations.csv', index_col=0)
    if not os.path.isfile(inputpath + '/local_permutations.csv') or overwrite:
        getlocal=True
    else:
        getlocal=False
        local_perms = pd.read_csv(inputpath + '/local_permutations.csv', index_col=0)
    if getglobal:
        global_dfs = [pd.read_csv(p, index_col=0) for p in global_permutations]
        global_permdf = pd.concat(global_dfs,ignore_index=True)
        global_perms = global_permdf.loc[:, ['threshold', 'permutation', 'average_clustering', 'average_shortest_path_length', 'assortativity', 'modularity', 'efficiency']]
        global_perms.to_csv(inputpath + '/global_permutations.csv')
    if getlocal:
        local_dfs = [pd.read_csv(p, index_col=0) for p in local_permutations]
        local_permdf = pd.concat(local_dfs,ignore_index=True)
        local_perms = local_permdf.loc[:, ['name', 'threshold', 'permutation', 'degree', 'clustering', 'closeness', 'betweenness', 'shortest_path_length', 'local_efficiency']]
        local_perms.to_csv(inputpath + '/local_permutations.csv')
    return(global_perms, local_perms)

def maxNull(perms, orig):
    # Function to find critical values as well as return the null distribution of the maximum value across thresholds
    # orig is a column of values for the original data
    # just needed to decide on which direction the max effect is in
    # permdf contains the permutation, threshold, and values for the permuted effect
    # find and return a single column with the max for each permutation
    metric = orig.name
    null_df = pd.DataFrame(columns=[metric])
    crit_df = pd.DataFrame(columns=[metric])
    # if original observed difference contains positive values
    if orig.max() > 0:
        # if it's only positive values
        if orig.min() >= 0:
            perm_maxdf = perms.groupby('permutation').max()
            null_df = perm_maxdf.loc[:, metric]
            crit_df = null_df.quantile(.95) #fyi this uses a linear interpolation for intermediate values
        # if it also has negative values
        elif orig.min() < 0:
            perm_maxdf = perms.abs().groupby('permutation').max()
            null_df = perm_maxdf.loc[:, metric]
            crit_df = null_df.quantile(.975) # two-sided test
    # if observed difference only contains negative values
    elif orig.max() <= 0:
        perm_maxdf = perms.groupby('permutation').min()
        null_df = perm_maxdf.loc[:, metric]
        crit_df = null_df.quantile(.05) # opposite side of the distribution
    return(null_df, crit_df)

def getCrit(df, permdf, level):
    # Get the maximum null distribution and critical threshold for each metric
    if level=='global':
        metrics = df.drop(columns=['threshold', 'module_nmi', 'wiring_cost_U', 'wiring_cost_p']).columns
        nulldf = pd.DataFrame()
        critdf = pd.DataFrame()
        for m in metrics:
            nulldf.loc[:, m], critdf.loc[0, m] = maxNull(permdf, df.loc[:, m])
    elif level=='local':
        metrics = df.drop(columns=['name', 'index', 'threshold']).columns
        nulldf = pd.DataFrame()
        critdf = pd.DataFrame(columns=metrics)
        for m in metrics:
            tmp_permdf = permdf.drop(columns=['name', 'index'])
            tmp_origdf = df.loc[:, m]
            tmp_nulldf, tmp_critdf = maxNull(tmp_permdf, tmp_origdf)
            tmp_nulldf = tmp_nulldf.to_frame()
            nulldf = pd.concat([nulldf, tmp_nulldf], axis=1)
            critdf.loc[0,m]=tmp_critdf
    return(nulldf, critdf)

def calcP(outputdf, nulldf, level, nPermutations=1000):
    # add a p-value to the critical output df based on maximum null distribution across thresholds
    pVal = []
    for row in range(len(outputdf)):
        stat = outputdf.loc[row, :]
        if level=='local':
            nullstat = nulldf.loc[nulldf.name==stat[0], stat.metric]
            #pstat = np.sum(nulldf.loc[nulldf.name==stat[0], stat.metric]>=stat.obsVal)/nPermutations
        elif level=='global':
            nullstat = nulldf.loc[:, stat.metric]
            #pstat = np.sum(nulldf.loc[:, stat.metric]>=stat.obsVal)/nPermutations
        if stat.obsVal > 0 and stat.critVal > 0:
            pstat = np.sum(nullstat>=stat.obsVal)/nPermutations
        elif stat.obsVal < 0 and stat.critVal < 0:
            pstat = np.sum(nullstat<=stat.obsVal)/nPermutations
        else:
            pstat = np.sum(nullstat.abs()<=abs(stat.obsVal))/nPermutations
        pVal = np.append(pVal, pstat)
    outputdf.loc[:, 'pVal'] = pVal
    return(outputdf)

def calcAUC(outputdf, df, permdf, level, maxThr=40, nInterp=500, nPerms=1000):
    # calculate the cluster AUC for each significant difference
    outputdf = outputdf.assign(obsAUC=np.nan, critAUC=np.nan)
    thresholds = range(5, maxThr+1, 1)
    for row in range(len(outputdf)):
        stat = outputdf.loc[row, :]
        if level=='local':
            node=stat['name']
            metric = stat.metric
            obsval = df.loc[df.name==node, metric]
            critval = stat.critVal
            pivot_permdf = permdf.loc[permdf.name==node, ['threshold','permutation', metric]].pivot(index='threshold', columns='permutation', values=metric).astype(float)
            # only use absolute values for permutations if two-sided testing
            if not ((obsval<0).all() | (obsval>0).all()):
                obsval = df.loc[df.name==node, metric].abs()
                critval = abs(stat.critVal)
                # pivot from long to wide, each column is thresholds at a specific permutation
                pivot_permdf = permdf.loc[permdf.name==node, ['threshold','permutation', metric]].pivot(index='threshold', columns='permutation', values=metric).abs().astype(float)
        elif level=='global':
            # select just the metric
            metric  = stat.metric
            obsval = df.loc[:, metric]
            critval = stat.critVal
            pivot_permdf = permdf.loc[:, ['threshold','permutation', metric]].pivot(index='threshold', columns='permutation', values=metric).astype(float) # interp won't work with object dtype
            if not ((obsval<0).all() | (obsval>0).all()):
                obsval = df.loc[:, metric].abs()
                critval = abs(stat.critVal)
                pivot_permdf = permdf.loc[:, ['threshold','permutation', metric]].pivot(index='threshold', columns='permutation', values=metric).abs().astype(float)
        # interpolate to a high-res grid like Mark did, default is only 500
        thresh_interp=np.linspace(start=np.min(thresholds), stop=np.max(thresholds), num=nInterp)
        obs_interp = np.interp(thresh_interp, thresholds, obsval)
        obs_interp -= critval
        pivot_permdf_interp = pivot_permdf.reindex(df.index.union(thresh_interp)).interpolate('values').loc[thresh_interp]
        pivot_permdf_interp = pivot_permdf_interp.subtract(critval)
        # if original effect was negative threshold to only keep negative values
        if (obsval<0).all():
            obs_interp[obs_interp>0] = 0
            pivot_permdf_interp[pivot_permdf_interp>0] = 0
        else:
            obs_interp[obs_interp<0] = 0
            pivot_permdf_interp[pivot_permdf_interp<0] = 0
        # get AUC for each permutation
        obsAUC = np.trapz(x=thresh_interp, y=obs_interp)
        critAUC = 0 # update each iteration
        for p in range(nPerms): # fix nPerms here
            critAUC += np.trapz(x=pivot_permdf_interp.index, y=pivot_permdf_interp.iloc[:, p])
        critAUC /= nPerms
        outputdf.loc[row, ['obsAUC', 'critAUC']] = obsAUC, critAUC
    return(outputdf)

def addOrig(df, g0_df, g1_df, grouping='alc'):
    # add original values to AUC csv file
    # needs group dfs as well as df of differences, fixed to suit my data
    # loop through rows and use name, threshold, metric combo to find relevant original values
    for row in range(df.shape[0]):
        node = df.loc[row, 'name']
        metric  = df.loc[row, 'metric']
        threshold = df.loc[row, 'threshold']
        if grouping=='alc':
            df.loc[row, 'hcVal'] = g0_df.loc[(g0_df.name==node) & (g0_df.threshold==threshold),metric].values
            df.loc[row, 'alcVal'] = g1_df.loc[(g1_df.name==node) & (g1_df.threshold==threshold),metric].values
        elif grouping=='alc+':
            df.loc[row, 'alcVal'] = g0_df.loc[(g0_df.name==node) & (g0_df.threshold==threshold),metric].values
            df.loc[row, 'alcpVal'] = g1_df.loc[(g1_df.name==node) & (g1_df.threshold==threshold),metric].values
    return(df)

def findHubs(df, maxThr=None):
    if maxThr is not None:
        thresholds = range(5, maxThr+1, 1)
        for t in thresholds:
            df.loc[df.threshold==t, 'd_hub'] = np.where(df.loc[df.threshold==t, 'degree'] > np.quantile(df.loc[df.threshold==t, 'degree'].values, 0.8), 1, 0)
            df.loc[df.threshold==t, 'b_hub'] = np.where(df.loc[df.threshold==t, 'betweenness'] > np.quantile(df.loc[df.threshold==t, 'betweenness'].values, 0.8), 1, 0)
            df.loc[df.threshold==t, 'c_hub'] = np.where(df.loc[df.threshold==t, 'closeness'] > np.quantile(df.loc[df.threshold==t, 'closeness'].values, 0.8), 1, 0)
            df.loc[df.threshold==t, 'hub'] = df.loc[:, ['d_hub', 'b_hub', 'c_hub']].sum(axis=1)
            df.loc[(df.threshold==t) & (df.participation_coefficient==0), 'hub'] = 0
    else:
        df.loc[:, 'd_hub'] = np.where(df.loc[:, 'degree'] > np.quantile(df.loc[:, 'degree'].values, 0.8), 1, 0)
        df.loc[:, 'b_hub'] = np.where(df.loc[:, 'betweenness'] > np.quantile(df.loc[:, 'betweenness'].values, 0.8), 1, 0)
        df.loc[:, 'c_hub'] = np.where(df.loc[:, 'closeness'] > np.quantile(df.loc[:, 'closeness'].values, 0.8), 1, 0)
        df.loc[:, 'hub'] = df.loc[:, ['d_hub', 'b_hub', 'c_hub']].sum(axis=1)
        df.loc[df.participation_coefficient==0, 'hub'] = 0
        df.loc[:, 'plothub'] = np.where(df.loc[:, 'hub'] >= 2, 2, 1)
    return(df)

def attack(G, method='random', iterations=None, index=None):
    # attack script using either random or targeted attack
    # input is a graph
    # returns a vector or dataframe with values
    if method=='targeted' and index is None:
        print("No index is given for targeted attack")
        return
    n_iterations=1 # set to 1 unless defined during call for random attacks
    if method=='random':
        if iterations is not None:
            n_iterations = iterations
        else:
            print("Only running %i random attack" % n_iterations)
    # get the number of nodes
    n_nodes = len(list(G.nodes))
    # define output variables to keep track of number of remaining nodes (G_n) and global efficiency of the graph (G_ge)
    G_n =  np.zeros(n_nodes)
    G_ge = np.zeros((n_nodes, n_iterations))
    for i in range(n_iterations):
        if method=='random':
            index =  np.random.permutation(n_nodes) # randomly pick a node
        G_tmp = G.copy()
        for idx, n in enumerate(index):
            # i==0 is intact graph before removing node n
            # ge reaches zero before we run out of nodes so missing the last n doesn't matter here
            G_ge[idx, i] = nx.global_efficiency(G_tmp)
            # only set surviving nodes once at it's a repeat
            if i==0:
                G_n[idx] = n_nodes - len(list(G_tmp.nodes))
            G_tmp.remove_node(n)
    return(G_ge, G_n)
