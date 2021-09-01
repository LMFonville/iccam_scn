import os
import glob

import numpy as np
import networkx as nx
import scona as scn
import pandas as pd
from scipy import stats
import pingouin as pg

import nibabel.freesurfer.mghformat as mgh
from nilearn.plotting import find_parcellation_cut_coords
from sklearn.metrics.cluster import adjusted_mutual_info_score
from neuroCombat import neuroCombat

def loadFS(fspath, measure, atlas, subcortical=True):
    # load freesurfer files
    if atlas=='DKT':
        parc = 'aparc'
    elif atlas=='Destrieux':
        parc = 'a2009s'
    if measure=='volume':
        aparcLH=fspath + '/vol_lh_' + parc + '.txt'
        aparcRH=fspath + '/vol_rh_' + parc + '.txt'
    elif measure=='thickness':
        aparcLH=fspath + '/ct_lh_' + parc + '.txt'
        aparcRH=fspath + '/ct_rh_' + parc + '.txt'
        # assumes a separate file for global thickness in fspath, I calculated this from thickness and surface area measures
        # calculation is from Freesurfer FAQ: bh.thickness = ( (lh.thickness * lh.surfarea) + (rh.thickness * rh.surfarea) ) / (lh.surfarea + rh.surfarea)
        # gCT should be a 2-by-n dataframe with a column for subject ID and one for global thickness for n subjects
        gCT=fspath + '/global_thickness_' + parc + '.txt'
    lh = pd.read_csv(aparcLH, sep= "\t")
    rh = pd.read_csv(aparcRH, sep= "\t")
    lh.rename(columns={lh.columns[0]: "ID"}, inplace = True) #fixed because I know it's the first column that contains ID values
    rh.rename(columns={rh.columns[0]: "ID"}, inplace = True)
    # get aparc coordinates
    aparcCentroids = fspath + 'centroids_' + parc + '_xyz.txt'
    aparcLabels = fspath + 'centroid_' + parc + '_labels.txt'
    centroids = np.loadtxt(aparcCentroids)
    labels = np.loadtxt(aparcLabels, delimiter = " ", dtype=str)
    centroids_df = pd.DataFrame(centroids, columns=['X', 'Y', 'Z'])
    centroids_df.loc[:, 'label'] = labels
    centroids_df.loc[:, 'label'] = centroids_df['label'].str.replace('.', '_')
    # merge hemispheres
    aparc = pd.merge(lh, rh)
    if measure=='thickness':
        aparc = aparc.drop(aparc.filter(like='MeanThickness').columns, axis=1) # drop this and use the global one calculated using thickness and area
        thickness=pd.read_csv(gCT, sep= "\t")
        aparc = pd.merge(aparc, thickness)
        # load global thickness and merge with demo
    regions=list(aparc.filter(like='_' + measure, axis=1).columns.values)
    regions_df = pd.DataFrame(regions, columns=['region'])
    if measure=='volume':
        regions_df.loc[:, 'label'] = regions_df['region'].str.replace('_volume', '').str.replace('&', '_and_')
    elif measure=='thickness':
        regions_df.loc[:, 'label'] = regions_df['region'].str.replace('_thickness', '').str.replace('&', '_and_')
    # sort regions based on coordinates (L-R, I-S, P-A)
    #FS_df = pd.concat([regions_df.reset_index(drop=True), centroids_df], axis=1)
    FS_df = regions_df.merge(centroids_df, on='label')
    split = FS_df["region"].str.split("_", n = 2, expand = True)
    FS_df['hemi'] = split[0]
    FS_df['name'] = split[2]
    lh_index = FS_df.loc[FS_df.hemi=='lh',:].sort_values(['X', 'Z', 'Y'], ascending=[True, True, True]).index.values
    rh_index = lh_index + len(FS_df)/2
    index = np.concatenate((lh_index, rh_index))
    FS_df = FS_df.loc[index,:].reset_index()
    col_idx=FS_df.index.values
    region_idx = FS_df.region.values
    # other columns to keep after sorting
    if measure=='volume':
        other_idx = ['ID', 'eTIV']
    elif measure=='thickness':
        other_idx = ['ID', 'globalThickness']
    # if subcortical regions included, load these before putting coords together
    if subcortical:
        # subcortical coordinates
        asegmri = os.path.join(fspath + 'fsaverage/mri/aseg.mgz')
        aseg_file = mgh.load(asegmri)
        coords, labels = find_parcellation_cut_coords(aseg_file, return_label_names=True)
        aseg_l_idx = [labels.index(l) for l in [10, 11, 12, 13, 17, 18, 26]] # based on aseg lookup table
        aseg_r_idx = [labels.index(r) for r in [49, 50, 51, 52, 53, 54, 58]] # based on aseg lookup table
        aseg_coord_df = pd.DataFrame(coords)
        aseg_coord_df.columns = ['X', 'Y', 'Z']
        # load files
        asegfile= os.path.join(fspath, 'aseg_out_new.txt')
        aseg = pd.read_csv(asegfile, sep="\t")
        subvolumes = ['ID',
                      'Left-Thalamus-Proper', 'Left-Caudate', 'Left-Putamen', 'Left-Pallidum', 'Left-Hippocampus', 'Left-Amygdala', 'Left-Accumbens-area',
                      'Right-Thalamus-Proper', 'Right-Caudate', 'Right-Putamen','Right-Pallidum', 'Right-Hippocampus', 'Right-Amygdala', 'Right-Accumbens-area']
        aseg.rename(columns={"Measure:volume": "ID"}, inplace = True)
        aseg = aseg.loc[:, subvolumes]
        # put cortical and subcortical coordinates together after sorting
        coords_df = FS_df.loc[FS_df.X<0, ['X', 'Y', 'Z']] # left cortical
        coords_df = coords_df.append(aseg_coord_df.loc[aseg_l_idx]) # left subcortical
        coords_df = coords_df.append(FS_df.loc[FS_df.X>0, ['X', 'Y', 'Z']]) # right cortical
        coords_df = coords_df.append(aseg_coord_df.loc[aseg_r_idx]) # right subcortical
        #coords = coords_df.values # puts them together as a tuple
        aparc=pd.concat([aparc[other_idx].reset_index(drop=True), aparc[region_idx]], axis=1)
        all_lh = aparc.merge(aseg).filter(regex='lh_|Left-', axis=1)
        all_rh = aparc.merge(aseg).filter(regex='rh_|Right-', axis=1) # this catches entorhinal as well so need to remove duplicates
        aparc = pd.concat([aparc[other_idx].reset_index(drop=True), all_lh, all_rh], axis=1)
    else:
        coords_df = FS_df.loc[:, ['X', 'Y', 'Z']]
        #coords = coords_df.values # puts them together as a tuple
        aparc=pd.concat([aparc[other_idx].reset_index(drop=True), aparc[region_idx]], axis=1)
    return(aparc, coords_df)

def loadDemo(labelfile, grouping='alc'):
    # load demographics data
    df = pd.read_csv(labelfile)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')] # get rid of any weirdly formatted extra columns
    df['ID'] = df['study_id'] # make ID variable, will amend later to match the imaging format
    # add site as a variable (1 is London, 2 is Manchester, 3 is Cambridge)
    df.loc[df.study_id > 3000, 'site'] = int(3)
    df.loc[df.study_id < 3000, 'site'] = int(2)
    df.loc[df.study_id < 2000, 'site'] = int(1)
    df.rename(columns={'wtar_iq_score': 'IQ'}, inplace=True)
    # grouping is default to alcohol-control for now but can change this later to other groupings
    if grouping=='alc':
        # alcohol dependence (1) and controls (0) only
        conditions = [
            (df.final_grouping==1),
            (df.final_grouping==2) & (df.alcohol_dependence==1),
            (df.final_grouping==3)
        ]
        labelling = [int(0), int(1), int(1)]
        #df['group'] = np.select(conditions, labelling, default=-1)
    elif grouping=='alc+':
        # alcohol dependence (0) and alcohol polydrug (1) only
        conditions = [
            (df.final_grouping==2) & (df.alcohol_dependence==1),
            (df.final_grouping==3)
        ]
        labelling = [int(1), int(0)]
    else:
        return
    df['group'] = np.select(conditions, labelling, default=-1)
    # remove motion and abstinence outliers and those with different T1w sequence
    outliers = [1037, 1041, 3016, 3021, 3007, 3009, 3011, 3038]
    df = df[~df.study_id.isin(outliers)]
    # make two dummy variables to use as covariates if using OLS only
    df['siteCam'] = df['site'].apply(lambda x: 1 if x == 2 else 0)
    df['siteMan'] = df['site'].apply(lambda x: 1 if x == 3 else 0)
    # only select the columns we want to use before dropping NAs
    columns = ['ID', 'group', 'site', 'siteCam', 'siteMan', 'age', 'IQ']
    df = df[df.columns.intersection(columns)]
    df = df.dropna()
    df['ID'] = df['ID'].astype(int)
    # add 'sub-' to match the imaging naming format I used
    df['ID'] = 'sub-' + df.ID.map(str)
    return(df)

def confounding(df, measure='volume', method='combat_ols'):
    regions = list(df.filter(regex='lh|Left|rh|Right', axis=1).columns.values)
    if method=='combat':
        # first adjust imaging measures for different scanners
        if measure=='volume':
            df_aparc = df.filter(regex='lh|Left|rh|Right|eTIV', axis=1)
        elif measure=='thickness':
            df_aparc = df.filter(regex='lh|Left|rh|Right|globalThickness', axis=1)
        covars = ['group', 'age', 'IQ', 'site'] # make sure not to remove effects of group, age or IQ when adjusting site
        covar_df = df.loc[:, covars]
        continuous_cols = ['age', 'IQ']
        batch_col = 'site'
        categorical_cols='group'
        data_combat = neuroCombat(df_aparc.T,
                                  covars=covar_df,
                                  batch_col=batch_col,
                                  categorical_cols=categorical_cols,
                                  continuous_cols=continuous_cols)
        combat_df = pd.DataFrame(data_combat.T, columns=df_aparc.columns)
        df_res = combat_df
    if method=='combat_ols':
        # first adjust imaging measures for different scanners
        if measure=='volume':
            df_aparc = df.filter(regex='lh|Left|rh|Right|eTIV', axis=1)
        elif measure=='thickness':
            df_aparc = df.filter(regex='lh|Left|rh|Right|globalThickness', axis=1)
        covars = ['group', 'age', 'IQ', 'site'] # make sure not to remove effects of group, age or IQ when adjusting site
        covar_df = df.loc[:, covars]
        continuous_cols = ['age', 'IQ']
        batch_col = 'site'
        categorical_cols='group'
        data_combat = neuroCombat(df_aparc.T,
                                  covars=covar_df,
                                  batch_col=batch_col,
                                  categorical_cols=categorical_cols,
                                  continuous_cols=continuous_cols)
        combat_df = pd.DataFrame(data_combat.T, columns=df_aparc.columns)
        # regress out effects of age and IQ (after scaling variables)
        combat_df = pd.concat([df.loc[:, ['ID', 'group', 'age', 'IQ']].reset_index(), combat_df], axis=1)
        combat_df.loc[:, 'IQs'] = (combat_df.IQ - np.mean(combat_df.IQ))/np.std(combat_df.IQ)
        combat_df.loc[:, 'ages'] = (combat_df.age - np.mean(combat_df.age))/np.std(combat_df.age)
        if measure=='volume':
            combat_df.loc[:, 'eTIVs'] = (combat_df.eTIV - np.mean(combat_df.eTIV))/np.std(combat_df.eTIV)
            covars = ['ages', 'IQs', 'eTIVs']
        elif measure=='thickness':
            combat_df.loc[:, 'globalCTs'] = (combat_df.globalThickness - np.mean(combat_df.globalThickness))/np.std(combat_df.globalThickness)
            covars = ['ages', 'IQs', 'globalCTs']
        df_res = scn.create_residuals_df(combat_df, regions, covars)
    elif method=='ols':
        # scale other covariates
        if measure=='volume':
            covars = ['ages', 'IQs', 'siteCam', 'siteMan', 'eTIVs']
            df.loc[:, 'eTIVs'] = (df.eTIV - np.mean(df.eTIV))/np.std(df.eTIV)
        elif measure=='thickness':
            covars = ['ages', 'IQs', 'siteCam', 'siteMan', 'globalCTs']
            df.loc[:, 'globalCTs'] = (df.globalThickness - np.mean(df.globalThickness))/np.std(df.globalThickness)
        df.loc[:, 'IQs'] = (df.IQ - np.mean(df.IQ))/np.std(df.IQ)
        df.loc[:, 'ages'] = (df.age - np.mean(df.age))/np.std(df.age)
        df_res = scn.create_residuals_df(df, regions, covars)
    return(df_res)

def buildDF(inputpath, atlas, fspath=None, labelfile=None, measure='volume', subcortical=True, grouping='alc', method=None):
    # build dataframes if files do not exist on disk
    if not os.path.isfile(inputpath + '/aparc.csv') or fspath is not None:
        aparc, coords = loadFS(fspath, measure, atlas=atlas, subcortical=subcortical)
        #save output for next time (naming convention assumes different atlases would be in different output folders)
        aparc.to_csv(inputpath + '/aparc.csv')
        coords.to_csv(inputpath + '/coords.csv')
    else:
        aparc = pd.read_csv(inputpath + '/aparc.csv')
        coords = pd.read_csv(inputpath + '/coords.csv')
    if not os.path.isfile(inputpath + '/demo.csv') or labelfile is not None:
        demo = loadDemo(labelfile, grouping=grouping)
        #save output for next time
        demo.to_csv(inputpath + '/demo.csv')
    else:
        demo = pd.read_csv(inputpath + '/demo.csv')
    # merge data and remove any rows not matching grouping
    df = pd.merge(demo, aparc)
    df = df[df.group!=-1].reset_index(drop=True)
    # adjust data for confounding variables
    if method is not None:
        df_res = confounding(df,measure=measure, method=method)
        df = pd.concat([df.loc[:, ['ID', 'group', 'site']].reset_index(drop=True), df_res], axis=1)
        filename = 'aparc_' + method + '.csv'
        df.to_csv(inputpath + os.sep + filename)
    return(df, coords)

def calcLocalEfficiency(G):
    # The nodal efficiency is the global efficiency computed on the neighborhood of a node
    local_efficiency = {}
    for node in G.nodes:
        neighbours = [n for n in G.neighbors(node)]
        # only calculate if node actually has neighbours
        if len(neighbours) > 2:
            SG=nx.subgraph(G, neighbours)
            local_efficiency[node] = nx.global_efficiency(SG)
        else:
            local_efficiency[node] = 0
    return(local_efficiency)

def calcNodalZ(G, df):
    # Modified from scona graph_measures.py using dataframe to create subgraphs instead
    # It does a z-score transformation of within module degree
    # Outputs align with participation coefficient scores
    G_modules = df.loc[:, 'module'].unique()
    for m in G_modules:
        z_score = {}
        idx = df.index[df.module==m]
        M = G.subgraph(idx)
        M_degrees = list(dict(M.degree()).values())
        M_degree = np.mean(M_degrees)
        M_std = np.std(M_degrees)
        for v in M.nodes:
            # Calculate the number of intramodule edges
            wm_edges = float(nx.degree(G=M, nbunch=v))
            # Calculate z score as the intramodule degree of v
            # minus the mean intramodule degree, all divided by
            # the standard deviation of intramodule degree
            if M_std != 0:
                zs = (wm_edges - M_degree)/M_std
            else:
                # If M_std is 0, then all M_degrees must be equal.
                # It follows that the intramodule degree of v must equal
                # the mean intramodule degree.
                # It is therefore valid to assign a 0 value to the z-score
                zs = 0
            z_score[v] = zs
            df.loc[v, 'z_score'] = zs
    return(df)

def getRobustMat(df, type='skipped'):
    # calculates and/or returns the correlation matrices
    # single file implementation that allows for different methods available in pingouin
    regions = list(df.filter(regex='lh|Left|rh|Right', axis=1).columns.values)
    n_regions = len(regions)
    # ensure input df has been split by group if applicable
    cordf = pg.pairwise_corr(df.loc[:, regions], method=type) # produces a long format dataframe that skips duplicates
    # use for loop to fill in both sides of triangle using dataframe
    # first region is n-1 long, second region is n-2 long, etc
    # this should leave the diagonal as containing zeros
    cormat = np.zeros((n_regions,n_regions))
    for i,r in enumerate(regions):
        cormat[i, i+1:] = cordf.loc[cordf.X==r, 'r'].values
        cormat[i+1:, i] = cordf.loc[cordf.X==r, 'r'].values
    # turn matrix into dataframe
    Mdf = pd.DataFrame(cormat)
    Mdf.columns = regions
    Mdf.index = regions
    return(Mdf)

def getGraphMeasures(M, coords, maxThr=40):
    # single matrix implementation, doesn't calculate group differences
    regions = list(M.filter(regex='lh|Left|rh|Right', axis=1).columns.values)
    coords = coords.values
    G = scn.BrainNetwork(network=M, parcellation=regions, centroids=coords)
    # prepare dataframes
    G_global_df = pd.DataFrame()
    G_local_df = pd.DataFrame()
    # collect graph measures across range of thresholds, minimum is fixed to 5%
    thresholds = range(5, maxThr+1, 1)
    # add calculation of local efficiency after node removal
    efficiency = {'local_efficiency': calcLocalEfficiency}
    # define variables for looking at nodal wiring cost
    wiring_cost = []
    p_val = []
    # calculate for each threshold
    for t in thresholds:
        # create graphs from correlation matrix
        Gt = G.threshold(t)
        # global metrics
        G_global = pd.DataFrame.from_dict(Gt.calculate_global_measures(), orient="index").T
        G_global.loc[0, 'threshold'] = t
        G_global_df = G0_global_df.append(G0_global, ignore_index=True)
        # local metrics
        Gt.calculate_nodal_measures(additional_measures = efficiency)
        scn.assign_nodal_distance(Gt) # add distance metrics to calculate a 'wiring cost'
        G_local = Gt.report_nodal_measures()
        G_local = calcNodalZ(Gt, G_local) # add z-score for within-module degree
        G_local.loc[:, 'threshold'] = t
        G_local_df = G_local_df.append(G_local, ignore_index=True)
        # average wiring cost comparison
        U, p = stats.mannwhitneyu(np.array(G0_local.average_dist.values, dtype='float'), np.array(G1_local.average_dist.values, dtype='float')) # set to float instead of object to avoid isnan error
        wiring_cost.append(U)
        p_val.append(p)
    # reorder columns local df
    G_local_df = G_local_df.loc[:, ['name', 'threshold', 'module', 'degree', 'clustering', 'closeness', 'betweenness', 'participation_coefficient', 'shortest_path_length', 'local_efficiency', 'z_score', 'average_dist', 'total_dist', 'x', 'y', 'z']]
    return(G_global_df, G_local_df)

def robustMat(inputpath, df, group='alc', overwrite=False, permutation=None):
    # calculates and/or returns the correlation matrices using skipped correlations
    # applied to my data looking at specific groups
    cor_path = inputpath + os.sep + 'correlations'
    if group=='alc':
        g0_name = 'HC_correlation'
        g1_name = 'ALC_correlation'
    elif group=='alc+':
        g0_name = 'ALC_correlation'
        g1_name = 'ALCp_correlation'
    if permutation is not None:
        g0_name = g0_name + '_perm' + str(permutation).zfill(4)
        g1_name = g1_name + '_perm' + str(permutation).zfill(4)
        cor_path = inputpath + os.sep + 'permutations/correlations'
    if not os.path.exists(cor_path):
        os.makedirs(cor_path)
    # if directory is empty or you want to overwrite
    if not os.listdir(cor_path) or overwrite:
        # use skipped correlations to estimate correlation coefficient instead
        regions = list(df.filter(regex='lh|Left|rh|Right', axis=1).columns.values)
        n_regions = len(regions)
        # group 0; healthy controls
        g0_cordf = pg.pairwise_corr(df.loc[df.group==0, regions], method='skipped') # produces a long format dataframe that skips duplicates
        # use for loop to fill in both sides of triangle using dataframe
        # first region is n-1 long, second region is n-2 long, etc
        # this should leave the diagonal as containing zeros
        mat0 = np.zeros((n_regions,n_regions))
        for i,r in enumerate(regions):
            mat0[i, i+1:] = g0_cordf.loc[g0_cordf.X==r, 'r'].values
            mat0[i+1:, i] = g0_cordf.loc[g0_cordf.X==r, 'r'].values
        #mat0 = vec_to_sym_matrix(g0_cordf.r, diagonal=np.ones(82)) # long df vector to symmetric matrix
        M0 = pd.DataFrame(mat0)
        M0.columns = regions
        M0.index = regions
        # group 1; alcohol dependents
        g1_cordf = pg.pairwise_corr(df.loc[df.group==1, regions], method='skipped') # produces a long format dataframe
        mat1 = np.zeros((n_regions,n_regions))
        for i,r in enumerate(regions):
            mat1[i, i+1:] = g1_cordf.loc[g1_cordf.X==r, 'r'].values
            mat1[i+1:, i] = g1_cordf.loc[g1_cordf.X==r, 'r'].values
        M1 = pd.DataFrame(mat1)
        M1.columns = regions
        M1.index = regions
        M0.to_csv(cor_path + os.sep + g0_name + '_matrix.csv')
        g0_cordf.to_csv(cor_path + os.sep + g0_name + '.csv')
        M1.to_csv(cor_path + os.sep + g1_name + '_matrix.csv')
        g1_cordf.to_csv(cor_path + os.sep + g1_name + '.csv')
    else:
        M0 = pd.read_csv(cor_path + os.sep + g0_name + '_matrix.csv', index_col=0) # avoid having rownames as separate column
        M1 = pd.read_csv(cor_path + os.sep + g1_name + '_matrix.csv', index_col=0)
    return(M0, M1)

def getSigma(Gt, t, nGraphs=1000):
    # calculate small-worldness for a single graph network at a single threshold
    G_bundle = scn.GraphBundle([Gt], ['G'])
    G_bundle.create_random_graphs('G', nGraphs)
    # report global metrics for random graphs (needed?)
    G_random_df = G_bundle.report_global_measures()
    # calculate small worldness
    G_sigma = G_bundle.report_small_world('G')
    G_sigma_df = pd.DataFrame.from_dict(G_sigma, orient='index', columns=['sigma'])
    # replace first entry with the mean
    G_sigma_df.loc['G', 'sigma'] = G_sigma_df.iloc[1:].mean().values
    # join with other data
    G_random_df = G_random_df.join(G_sigma_df)
    G_random_df.loc[:, 'threshold'] = t
    return(G_random_df)

def getSumAdjMat(G, regions, minThr=5, maxThr=40):
    # get adjacency matrix at each threshold and take the sum
    # e.g. value of 6 would mean present at each threshold from 5-10
    # using this for circular plots
    thresholds = range(minThr, maxThr+1, 1)
    adj_fullmat = None
    for t in thresholds:
        Gt = G.threshold(t) # binary graph
        adj_tmp = nx.adjacency_matrix(Gt).todense()
        if adj_fullmat is None:
            adj_fullmat = adj_tmp
        else:
            # store in long format to prep for plotting in R
            adj_fullmat += adj_tmp
    adj_df = pd.DataFrame(adj_fullmat, columns=regions, index=regions)
    return(adj_df)

def getRandomGraphs(M, threshold):
    regions = M.columns.values
    # reproduce the graph
    G = scn.BrainNetwork(network=M, parcellation=regions) # no need for coords to get the global metrics
    # threshold graph
    Gt = G.threshold(threshold)
    # produce random graphs and calculate sigma
    Gt_random = getSigma(Gt, threshold, 1000)
    return(Gt_random)

def DFtoGraphMeasuresRobust(df, coords, outputdir, maxThr=40, grouping='alc', permutation=None):
    # grouped implementation to get differences, also works with permutation testing
    # calculate correlation matrices and graph measures
    regions = list(df.filter(regex='lh|Left|rh|Right', axis=1).columns.values)
    coords = coords.values
    if permutation is not None:
        df = permuteDF(df, outputdir)
        G0_M, G1_M = robustMat(outputdir, df, group=grouping, overwrite=True, permutation=permutation)
    else:
        # alc: groups are coded as 0 for hc and 1 for alc
        # alc+: groups are coded as 0 for alc and 1 for alc+
        G0_M, G1_M = robustMat(outputdir, df, group=grouping)
    # prepare dataframes
    G0_global_df = pd.DataFrame()
    G0_local_df = pd.DataFrame()
    G1_global_df = pd.DataFrame()
    G1_local_df = pd.DataFrame()
    # collect graph measures across range of thresholds
    thresholds = range(5, maxThr+1, 1)
    efficiency = {'local_efficiency': calcLocalEfficiency}
    module_nmi = []
    wiring_cost = []
    p_val = []
    for t in thresholds:
        # create graphs from correlation matrices
        G0_G = scn.BrainNetwork(network=G0_M, parcellation=regions, centroids=coords)
        G0_Gt = G0_G.threshold(t)
        G1_G = scn.BrainNetwork(network=G1_M, parcellation=regions, centroids=coords)
        G1_Gt = G1_G.threshold(t)
        # global metrics
        G0_global = pd.DataFrame.from_dict(G0_Gt.calculate_global_measures(), orient="index").T
        G0_global.loc[0, 'threshold'] = t
        G0_global_df = G0_global_df.append(G0_global, ignore_index=True)
        G1_global = pd.DataFrame.from_dict(G1_Gt.calculate_global_measures(), orient="index").T
        G1_global.loc[0, 'threshold'] = t
        G1_global_df = G1_global_df.append(G1_global, ignore_index=True)
        # local metrics
        G0_Gt.calculate_nodal_measures(additional_measures = efficiency)
        scn.assign_nodal_distance(G0_Gt) # add distance metrics to calculate a 'wiring cost'
        G0_local = G0_Gt.report_nodal_measures()
        G0_local = calcNodalZ(G0_Gt, G0_local) # add z-score for within-module degree
        G0_local.loc[:, 'threshold'] = t
        G0_local_df = G0_local_df.append(G0_local, ignore_index=True)
        G1_Gt.calculate_nodal_measures(additional_measures = efficiency)
        scn.assign_nodal_distance(G1_Gt) # add distance metrics to calculate a 'wiring cost'
        G1_local = G1_Gt.report_nodal_measures()
        G1_local = calcNodalZ(G1_Gt, G1_local) # add z-score for within-module degree
        G1_local.loc[:, 'threshold'] = t
        G1_local_df = G1_local_df.append(G1_local, ignore_index=True)
        # modular agreement
        G0_modules = G0_local_df.loc[G0_local_df.threshold==t, 'module'].values
        G1_modules = G1_local_df.loc[G1_local_df.threshold==t, 'module'].values
        module_nmi.append(adjusted_mutual_info_score(G0_modules, G1_modules, average_method='arithmetic'))
        # average wiring cost comparison
        U, p = stats.mannwhitneyu(np.array(G0_local.average_dist.values, dtype='float'), np.array(G1_local.average_dist.values, dtype='float')) # set to float instead of object to avoid isnan error
        wiring_cost.append(U)
        p_val.append(p)
    # reorder columns
    G1_local = G1_local_df.loc[:, ['name', 'threshold', 'module', 'degree', 'clustering', 'closeness', 'betweenness', 'participation_coefficient', 'shortest_path_length', 'local_efficiency', 'z_score', 'average_dist', 'total_dist', 'x', 'y', 'z']]
    G0_local = G0_local_df.loc[:, ['name', 'threshold', 'module', 'degree', 'clustering', 'closeness', 'betweenness', 'participation_coefficient', 'shortest_path_length', 'local_efficiency', 'z_score', 'average_dist', 'total_dist', 'x', 'y', 'z']]
    # calculate differences
    # alc: HC minus ALC
    # alc+: ALC minus ALC+
    global_df = G0_global_df.sub(G1_global_df)
    global_df.loc[:, 'threshold'] = thresholds
    if permutation is None:
        global_df.loc[:, 'module_nmi'] = module_nmi
        global_df.loc[:, 'wiring_cost_U'] = wiring_cost
        global_df.loc[:, 'wiring_cost_p'] = p_val
    local_df = pd.concat([G1_local_df.loc[:, ['name', 'threshold']],
                          G0_local_df.drop(columns=['name', 'centroids', 'x', 'y', 'z', 'module', 'threshold']).sub(G1_local_df.drop(columns=['name', 'centroids', 'x', 'y', 'z', 'module', 'threshold']))], axis=1)
    # keep only metrics of interest for statistical comparisons
    local_df = local_df.loc[:, ['name', 'threshold', 'degree', 'clustering', 'closeness', 'betweenness', 'shortest_path_length', 'local_efficiency']]

    # check if this is for permutation testing or the original graphs
    if permutation is not None:
        outputdir = os.path.join(outputdir, 'permutations')
        #add permutation number to dataframe
        global_df.loc[:, 'permutation'] = permutation
        local_df.loc[:, 'permutation'] = permutation
    # change output path
    global_dir = os.path.join(outputdir, 'global')
    local_dir = os.path.join(outputdir, 'local')
    #random_dir = os.path.join(outputdir, 'random')
    if not os.path.exists(global_dir):
        os.makedirs(global_dir)
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    # store group outputs and move on
    if permutation is None:
        if grouping=='alc':
            G1_global_df.to_csv(global_dir + '/ALC_global.csv')
            G0_global_df.to_csv(global_dir + '/HC_global.csv')
            G1_local.to_csv(local_dir + '/ALC_local.csv')
            G0_local.to_csv(local_dir + '/HC_local.csv')
        elif grouping=='alc+':
            G0_global_df.to_csv(global_dir + '/ALC_global.csv')
            G1_global_df.to_csv(global_dir + '/ALCp_global.csv') # ALC+
            G0_local.to_csv(local_dir + '/ALC_local.csv')
            G1_local.to_csv(local_dir + '/ALCp_local.csv') # ALC+
        else:
            return
    if permutation is not None:
        # store group differences
        global_df.to_csv(global_dir + '/perm_' + str(permutation).zfill(4) + '_global_differences.csv')
        local_df.to_csv(local_dir + '/perm_' + str(permutation).zfill(4) + '_local_differences.csv')
        # store random graphs from permutations?
    else:
        global_df.to_csv(global_dir + '/global_differences.csv')
        local_df.to_csv(local_dir + '/local_differences.csv')
