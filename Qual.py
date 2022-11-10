import os
import numpy as np
import pandas as pd
import networkx as nx
from numpy import linalg
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

def eps_ts(ret_data, N, NF):
    """Generate noise time series with same covariance structure as the returns

       Arguments:
            ret_data: time series format dataframe (returns)
            N       : desired length of generated time series
            NF      : number of common factors (<= min(ret_data.shape))

       Returns:
            time series of noise (R tilde)
    """
    u, s, vt = np.linalg.svd(ret_data, full_matrices=False)  # 1a doc
    u_p = u[:, :NF]
    s_p = s[:NF]
    vt_p = vt[:NF, ]

    f_0 = (u_p * s_p) @ vt_p  # 1c doc

    r_tilde = ret_data.reset_index().iloc[:, 1:] - pd.DataFrame(f_0, columns=ret_data.columns)  # 1d doc

    rand2_df = pd.DataFrame(np.random.normal(0, 1, size=(len(ret_data.index), N)))

    eps = (np.dot(rand2_df.T, r_tilde)) / (np.sqrt(len(ret_data.index)))  # 2c doc
    uncor_df = pd.DataFrame(eps, columns=ret_data.columns)

    return uncor_df


def add_adj_noise(adj_mat, uncor_df, alpha, beta, lag):
    """Add adjacency effect to the time series (noise + common factors)

       Arguments:
            adj_mat    : adjacency matrix
            uncor_df   : time series for the adjacency effect to be applied on
            alpha      : ratio of the uncor_df to go through adjacency
            beta       : ratio of the uncor_df to stay without adjacency effect
            lag        : number of days (period) it takes the shock to spread

       Returns:
            time series with adjacency effect
    """
    noise_df = uncor_df*beta # pure noise
    adj_noise_df = uncor_df*alpha # input in adj

    adj_noise_df = adj_noise_df@(adj_mat[0]) # adj[0] to take the first time window only
    adj_noise_df.columns = noise_df.columns # rename columns to bbid's

    return adj_noise_df.shift(lag).fillna(0) + noise_df.values


def add_adj_noise_high_ord(adj_mat, uncor_df, alpha, beta, lag, itera):
    """Add the higher adjacency effect

       Arguments:
            adj_mat    : adjacency matrix
            uncor_df   : time series for the adjacency effect to be applied on
            alpha      : ratio of the uncor_df to go through adjacency
            beta       : ratio of the uncor_df to stay without adjacency effect
            lag        : number of days (period) it takes the shock to spread

       Returns:
            time series with adjacency effect
    """
    #data
    noise_df = uncor_df*beta # pure noise
    adj_noise_df = uncor_df*alpha # input for adj
    fin_adj_noise = pd.DataFrame(0, index = noise_df.index, columns = (noise_df.T).reset_index().T.columns)

    for ii in range(itera):
        adj_noise_df = adj_noise_df @ (adj_mat[0]) # adj[0] to take the first time window only
        fin_adj_noise = fin_adj_noise + adj_noise_df.shift(lag + ii).fillna(0)
        
    fin_adj_noise.columns = noise_df.columns # rename columns to bbid's
    return fin_adj_noise + noise_df.values


def cf_eps_ts_(ret_data, N, NF, adj_mat, alpha, beta, itera, lag_m):
    np.random.seed(30)
    """Add the higher adjacency effect

       Arguments:
            ret_data   : time series format dataframe (returns)
            adj_mat    : adjacency matrix
            uncor_df   : time series for the adjacency effect to be applied on
            alpha      : ratio of the uncor_df to go through adjacency
            beta       : ratio of the uncor_df to stay without adjacency effect
            itera      : higher order value of the adjacency spread (consdier the spread for how many days)
            lag_m      : number of days (period) it takes the shock to spread

       Returns:
            time series with adjacency effect
    """
    u, s, vt = np.linalg.svd(ret_data, full_matrices=False)  # 1a doc
    u_p = u[:, :NF]
    s_p = s[:NF]
    vt_p = vt[:NF, ]

    b = (vt_p.T * s_p).T  # 1b doc
    f_0 = (u_p * s_p) @ vt_p  # 1c doc
    r_tilde = ret_data.reset_index().iloc[:, 1:] - pd.DataFrame(f_0, columns=ret_data.columns)  # 1d doc

    rand1_df = pd.DataFrame(np.random.normal(0, 1, size=(len(ret_data.index), N)))
    l = (np.dot(rand1_df.T, u_p)) / (np.sqrt(len(ret_data.index)))  # 2a doc
    f_r = np.dot(l, b)  # 2b doc

    rand2_df = pd.DataFrame(np.random.normal(0, 1, size=(len(ret_data.index), N)))
    eps = pd.DataFrame((np.dot(rand2_df.T, r_tilde)) / (np.sqrt(len(ret_data.index))),
                       columns=ret_data.columns)  # 2c doc

    eps_ = add_adj_noise_high_ord(adj_mat, eps, alpha, beta, lag_m, itera)

    uncor_mat = f_r + eps_  # 3 doc
    uncor_df = pd.DataFrame(uncor_mat, columns=ret_data.columns)

    return uncor_df


def mle_b(data):
    """Compute the OLS for multivariate autoregression

       Arguments:
            data  : time series format data
            
       Returns:
            coefficinets of the model
    """

    x_m = data[:-1].values

    A = np.linalg.pinv(x_m.T@x_m)@x_m.T@data[1:]

    A.index = A.columns

    return A


def pcorr(mtx):
    """Compute the partial correlations

       Arguments:
            mtx: time series format dataframe

       Returns:
            partial correlations dataframe
       Note: This function is copied from the pingouin package (there were some incompatibilities with the python version and the package,
       couldn't install the package so just copied the function)
    """

    V = mtx.cov()  # Covariance matrix
    Vi = np.linalg.pinv(V)  # Inverse covariance matrix
    D = np.diag(np.sqrt(1 / np.diag(Vi)))
    pcor = -1 * (D @ Vi @ D)  # Partial correlation matrix
    pcor[np.diag_indices_from(pcor)] = 1
    return pd.DataFrame(pcor, index=V.index, columns=V.columns, dtype=object)


def pcor_lagged_nopv(ret_data, rw, lag1, d):
    """Compute the partial correlations for one time window

       Arguments:
            ret_data  : time series format data frame (returns)
            rw        : rolling window size
            lag1      : number of lags to consider in the partial correlation calculations
            d         : index to start from - set to 0 if want to consider the whole ret_data

       Returns:
            weighted adjacency dataframe
    """

    p_mat1 = pd.DataFrame(columns=ret_data.columns, dtype=object)

    for k in range(len(ret_data.columns)):
        sliced = ret_data.iloc[(d+lag1):(rw+d+lag1), :]
        tsn = ret_data.iloc[d:(rw+d), k].to_numpy()

        sliced.insert(loc=0, column="lagged", value=tsn)
        pcor_each = pcorr(sliced).iloc[0, 1:] # takes the partial cor of lead-lag variable with all the other ones
        pcor_each = pcor_each.reset_index()
            
        pcor_each = pcor_each.set_index("BBID")
        pcor_each.index.names = [None] # naming the rows by bbid
        pcor_each = pd.Series(pcor_each["lagged"])

        p_mat1.loc[k] = pcor_each # fill the matrix
        
    np.fill_diagonal(p_mat1.values, 0) # remove the loops
    return p_mat1


def cond_exp_arr(mtx):
    """Compute the conditional expectation array between lagged variabel and all other variables

       Arguments:
            mtx: time series format dataframe with lagged column

       Returns:
            conditional expectation 1Darray
    """

    V = mtx.cov()  # Covariance matrix
    Vi = np.linalg.pinv(V) # precision matrix
    cond_exp_ar = np.array([])
    
    for l in range(1, len(Vi)):
        cov = np.linalg.pinv([[Vi[0,0], Vi[0,l]], [Vi[l,0], Vi[l,l]]])  # Inverse of 2x2 precision matrix = cov
        cond_exp_v = cov[0,1] / cov[0,0] # conditional exp array bw lagged and others
        cond_exp_ar = np.append(cond_exp_ar, cond_exp_v)

    return cond_exp_ar


def cond_exp_lagged_df(ret_data, rw, lag1, d):
    """Compute the conditional expectation for one time window

       Arguments:
            ret_data  : time series format data frame (returns)
            rw        : rolling window size
            lag1      : number of lags to consider in the partial correlation calculations
            d         : index to start from - set to 0 if want to consider the whole ret_data

       Returns:
            weighted adjacency dataframe based on conditinoal expectation
    """
    
    p_mat1 = pd.DataFrame(columns = ret_data.columns, dtype=object)
    for k in range(len(ret_data.columns)):
        sliced = ret_data.iloc[(d+lag1):(rw+d+lag1), :]
        tsn = ret_data.iloc[d:(rw+d), k].to_numpy()

        sliced.insert(loc=0, column="lagged", value=tsn)
        pcor_each = cond_exp_arr(sliced) # takes the cond exp of lead-lag variable with all the other ones
        
        pcor_each = pd.DataFrame(pcor_each).set_index(ret_data.columns)
        pcor_each.index.names = [None] # naming the rows by bbid
        p_mat1.loc[k] = pd.Series(pcor_each[0]) # fill the matrix
        
    np.fill_diagonal(p_mat1.values, 0) # remove the loops
    return p_mat1


def leontief_inverse_node(adj_mat):
    """Get the Leontief inverse as centrality measure of the node

       Arguments:
            adj_mat: adjacency 3D array

       Returns:
            the Leontief inverse 3D array for adjacency matrix
    """

    leont = adj_mat.copy()
    #leont = adj_mat[:]
    for i in range(len(adj_mat)): # for each time window
        eig_mat = np.linalg.eig(adj_mat[i])[0]
        print(abs(max(eig_mat, key=abs)))

        iden = np.identity(len(adj_mat[0])) # identity matrix - number of nodes' size
        dif = iden - adj_mat[i]

        a_inv = np.linalg.inv(dif)
        leont[i][:] = a_inv
    return leont


def leontief_inverse_system(leon_node):
    """Get the Leontief inverse as centrality measure of the system

       Arguments:
            leon_node: Leontief inverse centrality 3D array

       Returns:
            Leontief inverse 2D array of adjacency matrix
    """

    return np.sum(leon_node, axis=-1) - 1 # -1 to exclude the initial shock


def mc_sim_adj_pcor(ret_data, N, NF, J, rw, lag_g, lag_m, adj_mat, alpha, beta, itera, d, cf):
    """Get adjacency based on partial correlation approach

       Arguments:
            ret_data: time series format dataframe (returns)
            N       : desired length of generated time series
            NF      : number of common factors (<= min(ret_data.shape))
            J       : number of simulations
            rw      : size of time window
            lag_g   : generation lag
            lag_m   : measurement lag
            adj_mat : adjacency 3D array
            alpha   : ratio of the uncor_df to go through adjacency
            beta    : ratio of the uncor_df to stay without adjacency effect
            itera   : higher order value of the adjacency spread (consdier the spread for how many days)
            d       : rolling the window (set to 0 at start)
            cf      : number of common factors (if no factors are considred input None)

       Returns:
            simulated adjacency based on partial correlation approach (dictionary)
    """
        
    np.random.seed(30)
    
    for i in range(J):
        if cf is None:
            sim_df = eps_ts(ret_data, N, NF)
            adj_df = add_adj_noise_high_ord(adj_mat, sim_df, alpha, beta, lag_g, itera)
        else:
            adj_df = cf_eps_ts_(ret_data, N, NF, adj_mat, alpha, beta, itera, lag_m)
            
        sim_pcor = pcor_lagged_nopv(adj_df, rw, lag_m, d)
        sim_n = sim_pcor.to_numpy()
        if i != 0:
            sim_fin = np.append(sim_fin, [sim_n], axis=0)
        else:
            sim_fin = [sim_n]

    fin_dict = {}
    fin_dict["sim_df"] = sim_fin

    return fin_dict


def mc_sim_adj_cond(ret_data, N, NF, J, rw, lag_g, lag_m, adj_mat, alpha, beta, itera, d, cf):
    """Get adjacency based on conditional expectation approach

       Arguments:
            ret_data: time series format dataframe (returns)
            N       : desired length of generated time series
            NF      : number of common factors (<= min(ret_data.shape))
            J       : number of simulations
            rw      : size of time window
            lag_g   : generation lag
            lag_m   : measurement lag
            adj_mat : adjacency 3D array
            alpha   : ratio of the uncor_df to go through adjacency
            beta    : ratio of the uncor_df to stay without adjacency effect
            itera   : higher order value of the adjacency spread (consdier the spread for how many days)
            d       : rolling the window (set to 0 at start)
            cf      : number of common factors (if no factors are considred input None)

       Returns:
            simulated adjacency based on conditional expectation approach (dictionary)
    """
        
    np.random.seed(30)
    
    for i in range(J):
        if cf is None:
            sim_df = eps_ts(ret_data, N, NF)
            adj_df = add_adj_noise_high_ord(adj_mat, sim_df, alpha, beta, lag_g, itera)
        else:
            adj_df = cf_eps_ts_(ret_data, N, NF, adj_mat, alpha, beta, itera, lag_m)
            
        sim_pcor = cond_exp_lagged_df(adj_df, rw, lag_m, d)
        sim_n = sim_pcor.to_numpy()
        
        if i != 0:
            sim_fin = np.append(sim_fin, [sim_n], axis = 0)
        else:
            sim_fin = [sim_n]

    fin_dict = {}
    fin_dict["sim_df"] = sim_fin

    return fin_dict


def mc_sim_adj_var(ret_data, N, NF, J, rw, lag_g, lag_m, adj_mat, alpha, beta, itera, d, cf):
    """Get adjacency based on multivariate autoregression approach

       Arguments:
            ret_data: time series format dataframe (returns)
            N       : desired length of generated time series
            NF      : number of common factors (<= min(ret_data.shape))
            J       : number of simulations
            rw      : size of time window
            lag_g   : generation lag
            lag_m   : measurement lag
            adj_mat : adjacency 3D array
            alpha   : ratio of the uncor_df to go through adjacency
            beta    : ratio of the uncor_df to stay without adjacency effect
            itera   : higher order value of the adjacency spread (consdier the spread for how many days)
            d       : rolling the window (set to 0 at start)
            cf      : number of common factors (if no factors are considred input None)

       Returns:
            simulated adjacency based on multivariate autoregression approach (dictionary)
    """
    
    np.random.seed(30)
    
    for i in range(J):        
        if cf is None:
            sim_df = eps_ts(ret_data, N, NF)
            adj_df = add_adj_noise(adj_mat, sim_df, alpha, beta, lag_g)
            
        else:
            adj_dff = cf_eps_ts_(ret_data, N, NF, adj_mat, alpha, beta, itera, lag_m)
            
            u_v, s_v, vt_v = np.linalg.svd(adj_dff, full_matrices = False)
            #plt.plot(s_v)
            
            u_v_p = u_v[:, :NF]
            s_v_p = s_v[:NF]
            vt_v_p = vt_v[:NF, ]

            f_0_v = (u_v_p*s_v_p)@vt_v_p # 1c doc
            
            adj_df = adj_dff - f_0_v
            #print(adj_df)
        
        sim_pcor = mle_b(adj_df)
        sim_n = sim_pcor.to_numpy()
        
        if i != 0:
            sim_fin = np.append(sim_fin, [sim_n], axis=0)
        else:
            sim_fin = [sim_n]

    fin_dict = {}
    fin_dict["sim_df"] = sim_fin

    return fin_dict


def leon_real_sim_plot(mc_sim, mode, adj_mat):
    """Plot system's leontief centrality for true and simulated time sereis

       Arguments:
            mc_sim : simulated dictionary
            mode   : network construction approach
            adj_mat: true adjacency matrix

       Returns:
            time series with adjacency effect
    """
    
    leon_mc_each = leontief_inverse_node(mc_sim["sim_df"])
    leon_mc_sys = leontief_inverse_system(leon_mc_each)
    
    plt.plot(np.mean(leon_mc_sys, axis = 0), label="Simulated", alpha=.7, color="C0")
    plt.plot(leontief_inverse_system(leontief_inverse_node([np.array(adj_mat)])).flatten(), label="Real", alpha=.7, color="C5")
    plt.xlabel("Node")
    plt.ylabel("Leontief centrality")
    plt.title(f"System's Leontief centrality for {mode} network")
    plt.legend();


def leon_real_sim_plot_normal(mc_sim, mode, adj_mat):
    """Plot system's leontief centrality for true and simulated time sereis - normalized

       Arguments:
            mc_sim : simulated dictionary
            mode   : network construction approach
            adj_mat: true adjacency matrix

       Returns:
            plot of normalized real and averaged simulated system Leontief centrality
    """
    
    leon_mc_each = leontief_inverse_node(mc_sim["sim_df"])
    leon_mc_sys = leontief_inverse_system(leon_mc_each)
    
    lm = np.mean(leon_mc_sys, axis = 0)
    lmn = lm/abs(max(lm, key=abs))
    
    ll = leontief_inverse_system(leontief_inverse_node([np.array(adj_mat)])).flatten()
    lln = ll/abs(max(ll, key=abs))
    
    plt.plot(lmn, label="Simulated", alpha=.7, color="C0")
    plt.plot(lln, label="Real", alpha=.7, color="C5")
    plt.xlabel("Node")
    plt.ylabel("Leontief centrality")
    plt.title(f"System's Leontief centrality for {mode} network")
    plt.legend();


def leon_real_sim_mse_normal(mc_sim, adj_mat):
    """Calculate MSE of the normalized simulated systme Leonteif centrality

       Arguments:
            mc_sim : simulated dictionary
            mode   : network construction approach
            adj_mat: true adjacency matrix

       Returns:
            MSE of nomalized system Leontief centrality
    """
    
    leon_mc_each = leontief_inverse_node(mc_sim["sim_df"])
    leon_mc_sys = leontief_inverse_system(leon_mc_each)
    
    lm = np.mean(leon_mc_sys, axis = 0)
    lmn = lm/abs(max(lm, key=abs))
    
    ll = leontief_inverse_system(leontief_inverse_node([np.array(adj_mat)])).flatten()
    lln = ll/abs(max(ll, key=abs))
    
    return (1/len(leon_mc_sys[-1]))*np.sum((lmn-lln)**2)#, axis = 0)


def leon_real_sim_mse_pval(mc_sim1, mc_sim2, adj_mat):
    """Calculate and compares MSE and p-value of the 25 and 0 factors simulated systme Leonteif centrality

       Arguments:
            mc_sim : simulated dictionary
            mode   : network construction approach
            adj_mat: true adjacency matrix

       Returns:
            MSE and p-value based on t-test
    """

    leon_mc_each1 = leontief_inverse_node(mc_sim1["sim_df"])
    leon_mc_sys1 = leontief_inverse_system(leon_mc_each1)
    
    leon_mc_each2 = leontief_inverse_node(mc_sim2["sim_df"])
    leon_mc_sys2 = leontief_inverse_system(leon_mc_each2)
    
    ll = leontief_inverse_system(leontief_inverse_node([np.array(adj_mat)])).flatten()

    mse1 = (1/len(leon_mc_sys1[-1]))*np.sum((leon_mc_sys1-ll)**2, axis = 1)
    mse2 = (1/len(leon_mc_sys2[-1]))*np.sum((leon_mc_sys2-ll)**2, axis = 1)
    
    return ttest_ind(mse1, mse2)


# same names throughout years
def all_years_same_name(raw_data, lag, ncf):
    """Calculate adjacency matrix from the data

       Arguments:
            raw_data : return data
            lag      : number of lags considered in partial cor and conditional exp
            ncf      : number of common factors considered

       Returns:
            derived adjacency matrix from data
    """

    kx = 0
    lin_dict  = {}
    cond_dict = {}
    pcor_dict = {}

    ret_cds_df_ = raw_data
    print(len(ret_cds_df_.columns))
    
    for ix in range(2005,2022): # each year
        ret_cds_df1 = ret_cds_df_.loc[f"{2005+kx}-01-01": f"{2005+kx}-12-31",:] # same names through years        
        rw = (len(ret_cds_df1)-lag)
        
        lin_mod = mle_b(ret_cds_df1)
        
        if len(ret_cds_df1) >= 252: # issues with pinv if ">(len(period))" in our case period = 252 days
            ret_cds_df_pcor = ret_cds_df1.iloc[:,:253]
            rw1 = (len(ret_cds_df_pcor)-lag)
            
            condexp_mod = cond_exp_lagged_df(ret_cds_df_pcor, rw1, lag, d = 0)
            pcor_mod = pcor_lagged_nopv(ret_cds_df_pcor, rw1, lag, d = 0)
            
        else:
            condexp_mod = cond_exp_lagged_df(ret_cds_df1, rw, lag, d = 0)
            pcor_mod = pcor_lagged_nopv(ret_cds_df1, rw, lag, d = 0)
            
        # model dict per year
        lin_dict[f"{2005+kx}-01-01", f"{2005+kx}-12-31"]  = lin_mod.to_numpy()
        cond_dict[f"{2005+kx}-01-01", f"{2005+kx}-12-31"] = condexp_mod.to_numpy()
        pcor_dict[f"{2005+kx}-01-01", f"{2005+kx}-12-31"] = pcor_mod.to_numpy()
        
        kx += 1 # go to next year
    
    # final dict per model (per year embedded in model already)
    fin_dict = {}
    fin_dict["adj_linear"]  = lin_dict
    fin_dict["adj_condexp"] = cond_dict
    fin_dict["adj_pcor"]    = pcor_dict

    return fin_dict


def sys_boxplot(adj_dict, adj_type):
    """boxplot of system's leontief centrality

       Arguments:
            adj_dict : dictionary obtained from the all_years_same_name function
            adj_type : type of network

       Returns:
            boxplot per year
    """

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    for kx in range(2022-2005):
        leont_nt = leontief_inverse_node(np.array([adj_dict[adj_type][f"{2005+kx}-01-01", f"{2005+kx}-12-31"]]))
        leo_syst = leontief_inverse_system(leont_nt)
        
        ax.boxplot(leo_syst.tolist(), positions = [kx], widths = 0.6, showmeans=True)
    plt.xlabel("year")
    plt.ylabel("Leontief centrality")
    plt.title(f"System's Leontief centrality for {adj_type} network")
    plt.xticks(np.arange(0,17), np.arange(2005, 2022), rotation=45);


def sys_boxplot_summary_stat(adj_dict, adj_type):
    """summary statistics of boxplot

       Arguments:
            adj_dict : dictionary obtained from the all_years_same_name function
            adj_type : type of network

       Returns:
            summary statistics per year
    """

    for kx in range(2022 - 2005):
        leont_nt = leontief_inverse_node(
            np.array([adj_dict[adj_type][f"{2005+kx}-01-01", f"{2005+kx}-12-31"]])
        )
        leo_syst = pd.DataFrame(leontief_inverse_system(leont_nt).flatten())
        print(leo_syst.describe())


def leontief_inverse_system_hist_markcap(leon_node, mark_cap, mx):
    """Get the historical market cap (end or average) weighted Leontief inverse as centrality measure of the system

       Arguments:
            leon_node: Leontief inverse centrality 3D array
            mark_cap : weight dataframe (market cap, liab/asset, ...)
            mx       : counter for years (determined through kx within other functions)

       Returns:
            Leontief inverse 1D array of adjacency matrix (the unweighted returns 2D array)
    """
    leon_node_2d = leon_node.reshape(len(mark_cap.columns),len(mark_cap.columns))
    return (leon_node_2d@np.array(mark_cap.T)[:,mx]) - 1


def sys_boxplot_hist_markcap(adj_dict, adj_type, mark_cap):
    """boxplot of system's leontief centrality

       Arguments:
            adj_dict : dictionary obtained from the all_years_same_name function
            adj_type : type of network
            mark_cap : weight dataframe 

       Returns:
            weighted boxplot per year
    """

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    for kx in range(2022-2005):
        leont_nt = leontief_inverse_node(np.array([adj_dict[adj_type][f"{2005+kx}-01-01", f"{2005+kx}-12-31"]]))
        leo_syst = leontief_inverse_system_hist_markcap(leont_nt, mark_cap, kx)
        #print(leo_syst)
        
        ax.boxplot(leo_syst.tolist(), positions = [kx], widths = 0.6, showmeans=True)#, patch_artist = True)
    plt.xlabel("year")
    plt.ylabel("Leontief centrality")
    plt.title(f"System's Leontief centrality for {adj_type} network")
    plt.xticks(np.arange(0,17), np.arange(2005, 2022), rotation=45);


def sys_boxplot_summary_stat_weight(adj_dict, adj_type, hist_var):
    """summary statistics of weighted boxplot

       Arguments:
            adj_dict : dictionary obtained from the all_years_same_name function
            adj_type : type of network
            hist_var : weight dataframe

       Returns:
            summary statistics per year
    """

    for kx in range(2022 - 2005):
        leont_nt = leontief_inverse_node(np.array([adj_dict["adj_linear"][f"{2005+kx}-01-01", f"{2005+kx}-12-31"]]))
        leo_syst = leontief_inverse_system_hist_markcap(leont_nt, hist_var, kx)
        leo_syst = pd.DataFrame(leo_syst)
        print(leo_syst.describe())

################ Outputs ################
os.chdir('/Users/stevensuser/Library/CloudStorage/OneDrive-stevens.edu/PhD_research/BB')
ret_cds_df = pd.read_pickle("ret_cds_us_20nan_5_21.pkl")
os.chdir('/Users/stevensuser/Library/CloudStorage/OneDrive-stevens.edu/PhD_research/Qual')
bb_info_all = pd.read_csv("qual_data_info_code_input.csv")
hist_liab = pd.read_csv("qual_data_hist_liab_ave_input.csv", index_col=0)
hist_asset = pd.read_csv("qual_data_hist_asset_ave_input.csv", index_col=0)
hist_liab_asset = hist_liab/hist_asset

## simulations
# 25 common factors (cf) link dist
adj_mat1 = mle_b(ret_cds_df.iloc[1040:1300,:100]) # 1040:1300 is corresponding to year 2009 data
mc_var_25cf = mc_sim_adj_var(ret_cds_df.iloc[1040:1300,:100], 252, 25, 10000, 251, 1, 1, [np.array(adj_mat1)], 1, 1, 50, 0, None)

clr = 1
for ix in [(1, 2), (1, 7), (5, 7), (100, 2)]:
    plt.hist(mc_var_25cf["sim_df"][:,ix[0]-1,ix[1]-1], bins=30, label=f"{ix[0]}:{ix[1]}", color=f"C{clr}", alpha=.5)
    plt.axvline(x = adj_mat1.iloc[ix[0]-1,ix[1]-1], color = f"C{clr}")
    clr += 1

plt.xlabel("Weight of links")
plt.ylabel("Frequency")
plt.title("Linear model network links distribution")
plt.legend();

# 25 cf bias
plt.scatter(np.array(adj_mat1).flatten(), np.mean(mc_var_25cf["sim_df"], axis=0).flatten())
plt.xlabel("Real adjacency")
plt.ylabel("Estimated adjacency")
plt.title("Adjacency matrix bias");

# 25 cf true vs sim system leont
leon_real_sim_plot(mc_var_25cf, "linear model", adj_mat1)

# 25 cf true vs sim system leont normalized
leon_real_sim_plot_normal(mc_var_25cf, "linear model", adj_mat1)

# 25 cf sytem leont mse
leon_real_sim_mse_normal(mc_var_25cf, adj_mat1)

# true vs sim time series for one name 25cf - this is gonna be different each time since generating 
# random ts in sumulation - only one simulation is done
# taking out 25 factors from the real ts first
u_v, s_v, vt_v = np.linalg.svd(ret_cds_df.iloc[1040:1300,:100], full_matrices = False)
u_v_p = u_v[:, :25]
s_v_p = s_v[:25]
vt_v_p = vt_v[:25, ]
f_0_v = (u_v_p*s_v_p)@vt_v_p
adj_df_real = ret_cds_df.iloc[1040:1300,:100].reset_index().iloc[:,1:] - pd.DataFrame(f_0_v, columns = ret_cds_df.iloc[:,:100].columns)

sim_df = eps_ts(ret_cds_df.iloc[1040:1300,:100], 252, 25)
adj_df = add_adj_noise([np.array(adj_mat1)], sim_df, 1, 1, 1)
plt.plot(adj_df.iloc[:,20].cumsum(), label = "Simulated")
plt.plot(adj_df_real.iloc[:252,20].reset_index(drop=True).cumsum(), label = "Real")
plt.xlabel("Time point")
plt.ylabel("Value")
plt.title("Real and simulated residuals time series")
plt.legend();

# true vs sim standard deviation for 25cf - only one simulation is done
plt.plot(np.std(adj_df, axis = 1), label="Simulated", alpha=.7)
plt.plot(np.std(adj_df_real.iloc[:252,:].reset_index(drop=True), axis=1), label="Real", alpha=.7)
plt.xlabel("Time point")
plt.ylabel("std")
plt.title("Standard deviation of real and simulated residuals time series")
plt.legend();

# no cf link dist
adj_mat1 = mle_b(ret_cds_df.iloc[1040:1300,:100])
mc_var_cf_0 = mc_sim_adj_var(ret_cds_df.iloc[1040:1300,:100], 252, 0, 10000, 251, 1, 1, [np.array(adj_mat1)], 1, 1, 50, 0, None)

clr = 1
for ix in [(1, 2), (1, 7), (5, 7), (100, 2)]:
    plt.hist(mc_var_cf_0["sim_df"][:,ix[0]-1,ix[1]-1], bins=30, label=f"{ix[0]}:{ix[1]}", color=f"C{clr}", alpha=.5)
    plt.axvline(x = adj_mat1.iloc[ix[0]-1,ix[1]-1], color = f"C{clr}")
    clr += 1

plt.xlabel("Weight of links")
plt.ylabel("Frequency")
plt.title("Linear model network links distribution")
plt.legend();

# no cf bias
plt.scatter(np.array(adj_mat1).flatten(), np.mean(mc_var_cf_0["sim_df"], axis=0).flatten())
plt.xlabel("Real adjacency")
plt.ylabel("Estimated adjacency")
plt.title("Adjacency matrix bias");

# no cf true vs sim system leont
leon_real_sim_plot(mc_var_cf_0, "linear model", adj_mat1)

# no cf true vs sim system leont normalized
leon_real_sim_plot_normal(mc_var_cf_0, "linear model", adj_mat1)

# no cf sytem leont mse
leon_real_sim_mse_normal(mc_var_cf_0, adj_mat1)

# true vs sim time series for one name no cf
sim_df = eps_ts(ret_cds_df.iloc[1040:1300,:100], 252, 0)
adj_df = add_adj_noise([np.array(adj_mat1)], sim_df, 1, 1, 1)
plt.plot(adj_df.iloc[:,20].cumsum(), label = "Simulated")
plt.plot(ret_cds_df.iloc[1040:1300,20].reset_index(drop=True).cumsum(), label = "Real")
plt.xlabel("Time point")
plt.ylabel("Value")
plt.title("Real and simulated residuals time series")
plt.legend();

# true vs sim standard deviation for no cf - only one simulation is done
plt.plot(np.std(adj_df, axis = 1), label="Simulated", alpha=.7)
plt.plot(np.std(ret_cds_df.iloc[1040:1300,:].reset_index(drop=True), axis=1), label="Real", alpha=.7)
plt.xlabel("Time point")
plt.ylabel("std")
plt.title("Standard deviation of real and simulated residuals time series")
plt.legend();

# 25 cf mse vs no cf mse
leon_real_sim_mse_pval(mc_var_25cf, mc_var_cf_0, adj_mat1) #p < 0.05, means are different

## real data
adj_same_name_test = all_years_same_name(raw_data = ret_cds_df, lag = 1, ncf=20)

# unweighted boxplot
sys_boxplot(adj_same_name_test, "adj_linear")

#names for unweighted outliers
for kx in range(2022-2005):
    leont_nt = leontief_inverse_node(np.array([adj_same_name_test["adj_linear"][f"{2005+kx}-01-01", f"{2005+kx}-12-31"]]))
    leo_syst = leontief_inverse_system(leont_nt)
    ind = np.argmax(leo_syst)
    print(leo_syst.max())
    print(bb_info_all.iloc[ind,:])

# unweighted summary statistics
sys_boxplot_summary_stat(adj_same_name_test, "adj_linear")

# weighted by historical liability to asset ratio boxplot
sys_boxplot_hist_markcap(adj_same_name_test, "adj_linear", hist_liab_asset)

#names for weighted by historical liability to asset outliers
for kx in range(2022-2005):
    leont_nt = leontief_inverse_node(np.array([adj_same_name_test["adj_linear"][f"{2005+kx}-01-01", f"{2005+kx}-12-31"]]))
    leo_syst = leontief_inverse_system_hist_markcap(leont_nt, hist_liab_asset, kx)
    ind = np.argmax(leo_syst)
    print(leo_syst.max())
    print(bb_info_all.iloc[ind,:])

# weighted historical liability to asset summary statistics
sys_boxplot_summary_stat_weight(adj_same_name_test, "adj_linear", hist_liab_asset)

#one name through time - weighted by historical liability to asset
plott = []
for kx in range(2022-2005):
    leont_nt = leontief_inverse_node(np.array([adj_same_name_test["adj_linear"][f"{2005+kx}-01-01", f"{2005+kx}-12-31"]]))
    leo_syst = leontief_inverse_system_hist_markcap(leont_nt, hist_liab_asset, kx)
    ind = bb_info_all[bb_info_all.TICKER == "JPM"].index
    plott = np.append(plott,leo_syst[ind])

plt.plot(plott/abs(max(plott, key=abs)))
plt.xlabel("year")
plt.ylabel("Leontief centrality")
plt.title("Normalized system's Leontief centrality for JPM through years")
plt.xticks(np.arange(0,17), np.arange(2005, 2022), rotation=45);

# financial sector data
fin_bb_id = bb_info_all[bb_info_all.BICS_LEVEL_1_SECTOR_NAME == "Financials"]["ID_BB_COMPANY"]
fin_index = bb_info_all[bb_info_all.BICS_LEVEL_1_SECTOR_NAME == "Financials"].index

adj_same_name_fin = all_years_same_name(raw_data = ret_cds_df[fin_bb_id], lag = 1, ncf=20)
sys_boxplot(adj_same_name_fin, "adj_linear") #unweighted fin

#names for unweighted fin outliers
for kx in range(2022-2005):
    leont_nt = leontief_inverse_node(np.array([adj_same_name_fin["adj_linear"][f"{2005+kx}-01-01", f"{2005+kx}-12-31"]]))
    leo_syst = leontief_inverse_system(leont_nt)
    ind = np.argmax(leo_syst)
    print(leo_syst.max())
    inf_fin = bb_info_all.iloc[fin_index,:].reset_index(drop = True)
    print(inf_fin.iloc[ind,:])

sys_boxplot_hist_markcap(adj_same_name_fin, "adj_linear", hist_liab_asset.iloc[:,fin_index])

#names for hist liab to asset finance outliers
for kx in range(2022-2005):
    leont_nt = leontief_inverse_node(np.array([adj_same_name_fin["adj_linear"][f"{2005+kx}-01-01", f"{2005+kx}-12-31"]]))
    leo_syst = leontief_inverse_system_hist_markcap(leont_nt, hist_liab_asset.iloc[:,fin_index], kx)
    ind = np.argmax(leo_syst)
    print(leo_syst.max())
    inf_fin = bb_info_all.iloc[fin_index,:].reset_index(drop = True)
    print(inf_fin.iloc[ind,:])

sys_boxplot_summary_stat_weight(adj_same_name_fin, "adj_linear", hist_liab_asset.iloc[:,fin_index])

#one name through time - liab/asset finance
plott = []
for kx in range(2022-2005):
    leont_nt = leontief_inverse_node(np.array([adj_same_name_fin["adj_linear"][f"{2005+kx}-01-01", f"{2005+kx}-12-31"]]))
    leo_syst = leontief_inverse_system_hist_markcap(leont_nt, hist_liab_asset.iloc[:,fin_index], kx)
    inf_fin = bb_info_all.iloc[fin_index,:].reset_index(drop = True)
    ind = inf_fin[inf_fin.TICKER == "BAC"].index
    
    plott = np.append(plott,leo_syst[ind])

plt.plot(plott/abs(max(plott, key=abs)))
plt.xlabel("year")
plt.ylabel("Leontief centrality")
plt.title("Normalized system's Leontief centrality for JPM through years")
plt.xticks(np.arange(0,17), np.arange(2005, 2022), rotation=45);

# visualization of financial network
kx = 15
A = adj_same_name_fin["adj_linear"][f"{2005+kx}-01-01", f"{2005+kx}-12-31"].copy()
print(A.max(),A.min(),np.mean(A))
A[abs(A) < 0.1] = 0
G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
pos = nx.circular_layout(G, scale = 100)
wt = nx.get_edge_attributes(G, 'weight')
plt.figure(figsize=(7,7))
plt.title(f"{2005+kx} Financials CDS return network")

nx.draw_networkx_edges(G, pos, width=list(wt.values()))
nx.draw_networkx_nodes(G, pos, node_color = "limegreen")
nx.draw_networkx_labels(G, pos, labels = bb_info_all.iloc[fin_index,1].reset_index(drop=True), font_size=10 ,verticalalignment = "center")
plt.show()

# visualization of network
y = 2020
A = adj_same_name_test['adj_linear'][f"{y}-01-01", f"{y}-12-31"].copy()
A
A = A[:30,:30]
A[abs(A) < 0.5] = 0

kx = 15
G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
pos = nx.circular_layout(G, scale = 100)
wt = nx.get_edge_attributes(G, 'weight')
plt.figure(figsize=(7,7))
plt.title(f"{2005+kx} CDS return network")
nx.draw_networkx_edges(G, pos, width=list(wt.values()))
nx.draw_networkx_nodes(G, pos, node_color = "limegreen")
nx.draw_networkx_labels(G, pos, labels = bb_info_all.iloc[:30,1], font_size=10 ,verticalalignment = "center")
plt.show()
