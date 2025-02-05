import numpy as np
from sklearn.impute import IterativeImputer
from sklearn.base import clone
from lifelines.statistics import multivariate_logrank_test



def kaplan_meier_weight(t_l, distinct_times, D, N):
    """
    Calculate the Kaplan-Meier weight for the Prentice-Wilcoxon test.
    """
    weight = 1.0
    for t_idx, t_prime in enumerate(distinct_times):
        if t_prime >= t_l:
            break
        weight *= 1 - (D[t_idx] / N[t_idx])
    return 1

def wilcoxon_prentice(Z, T, C, Delta):
    """
    Compute the Wilcoxon-Prentice test statistic.
    """
    R = np.minimum(T, C)
    distinct_times = np.sort(np.unique(R[Delta == 1]))

    N_treated, N_control, N_total = [], [], []
    D_treated, D_control, D_total = [], [], []

    for t in distinct_times:
        at_risk_treated = np.sum((Z == 1) & (R >= t))
        at_risk_control = np.sum((Z == 0) & (R >= t))
        at_risk_total = at_risk_treated + at_risk_control

        events_treated = np.sum((Z == 1) & (R == t) & (Delta == 1))
        events_control = np.sum((Z == 0) & (R == t) & (Delta == 1))
        events_total = events_treated + events_control

        N_treated.append(at_risk_treated)
        N_control.append(at_risk_control)
        N_total.append(at_risk_total)
        D_treated.append(events_treated)
        D_control.append(events_control)
        D_total.append(events_total)

    N_treated = np.array(N_treated)
    N_control = np.array(N_control)
    N_total = np.array(N_total)
    D_treated = np.array(D_treated)
    D_control = np.array(D_control)
    D_total = np.array(D_total)

    weights = np.array([
        kaplan_meier_weight(t, distinct_times, D_total, N_total)
        for t in distinct_times
    ])

    E_treated = (D_total * N_treated) / N_total
    A_W = np.sum(weights * (D_treated - E_treated))

    return A_W

def my_logrank_test(Z, T, C, Delta):
    """
    Perform a log-rank test between treatment groups using multivariate_logrank_test.

    Parameters:
        Z (array-like): Treatment assignment (1 = treatment, 0 = control).
        T (array-like): Realized event times.
        C (array-like): Censoring times.
        Delta (array-like): Event indicators (1 = event occurred, 0 = censored).

    Returns:
        p_value (float): The p-value of the log-rank test.
        test_statistic (float): The test statistic of the log-rank test.
    """
    # Ensure inputs are NumPy arrays for consistency
    Z = np.asarray(Z).ravel()  # Ensure Z is a flat 1D array
    T = np.asarray(T)
    C = np.asarray(C)

    # Compute observed times (minimum of event or censoring times)
    observed_times = np.minimum(T, C)

    # Perform the multivariate log-rank test
    results = multivariate_logrank_test(
        event_durations=observed_times,
        groups=Z,
        event_observed=Delta,
        weightings='wilcoxon'
    )

    return results.test_statistic

def preprocess_survival_data(Z, X, T, C, delta,missing_mask, G):
    """
    Impute missing values in delta using logistic regression (or specified classifier G).
    Return the original T and C without modification.
    """
    # Concatenate continuous features
    data_continuous = np.concatenate([Z.reshape(-1, 1), X], axis=1)  # No T and C
    missing_mask = missing_mask.astype(bool).flatten()  # Ensure it's boolean

    
    if G is None:
        delta_hat = delta
    else:
        # Impute missing delta values using the specified classifier (G)
        X_train = data_continuous[~missing_mask.flatten(), :]  # Rows where delta is NOT missing
        y_train = delta[~missing_mask.flatten()].ravel()  # Known delta values
        X_test = data_continuous[missing_mask.flatten(), :]  # Rows where delta is missing

        # Clone the classifier to avoid modifying original instance
        logistic_model = clone(G)
        logistic_model.fit(X_train, y_train)

        # Predict missing delta values
        delta_pred = logistic_model.predict(X_test)

        # Fill missing values
        delta[missing_mask] = delta_pred.reshape(-1, 1)
        
        delta_hat = delta.astype(int)  # Ensure binary output

    return T, C, delta_hat

def _preprocess_survival_data(Z, X, T, C, delta, G):
    """
    Impute missing T and C, and calculate Delta.
    """
    data = np.concatenate([Z.reshape(-1, 1), X, T.reshape(-1, 1), C.reshape(-1, 1), delta.reshape(-1,1)], axis=1)


    if G is None:
        imputed_data = data
    else:
        imputer = clone(G)
        imputed_data = imputer.fit_transform(data)    

    # Extract imputed delta
    delta_hat = imputed_data[:, -1].astype(int)

    return T, C, delta_hat

def getZsimTemplates(Z_sorted, S):
    """
    Create a Z_sim template for each unique value in S.
    """
    Z_sim_templates = []
    unique_strata = np.unique(S)
    for stratum in unique_strata:
        strata_indices = np.where(S == stratum)[0]
        strata_Z = Z_sorted[strata_indices]
        p = np.mean(strata_Z)
        strata_size = len(strata_indices)
        Z_sim_template = [0.0] * int(strata_size * (1 - p)) + [1.0] * int(strata_size * p)
        Z_sim_templates.append(Z_sim_template)
    return Z_sim_templates

def getZsim(Z_sim_templates):
    """
    Shuffle each Z_sim template and concatenate them into a single permutated Z_sim array.
    """
    Z_sim = []
    for Z_sim_template in Z_sim_templates:
        strata_Z_sim = np.array(Z_sim_template.copy())
        np.random.shuffle(strata_Z_sim)
        Z_sim.append(strata_Z_sim)
    Z_sim = np.concatenate(Z_sim).reshape(-1, 1)
    return Z_sim

def imputation_reimputation_survival(Z, X_star, T_star, C_star, S, delta,missing_mask, G, L=10000, randomization_design='strata', verbose=False):
    """
    iArt framework for survival data using the Wilcoxon-Prentice test statistic.
    """
    # Step 1: Impute missing T and C, and calculate observed test statistic
    T_hat, C_hat, Delta_hat = preprocess_survival_data(Z, X_star, T_star, C_star, delta, missing_mask,G)
    a = my_logrank_test(Z.ravel(), T_hat, C_hat, Delta_hat)#wilcoxon_prentice(Z.ravel(), T_hat, C_hat, Delta_hat)#my_logrank_test(Z.ravel(), T_hat, C_hat, Delta_hat)

    if verbose:
        print(f"Observed test statistic (a): {a}")

    # Step 2: Generate Z simulations and calculate test statistics for each
    test_statistics = []
    Z_templates = getZsimTemplates(Z, S)

    for l in range(L):
        Z_sim = getZsim(Z_templates)  # Simulate Z
        T_hat_sim, C_hat_sim, Delta_hat_sim = preprocess_survival_data(Z_sim, X_star, T_star, C_star, delta, missing_mask, G)
        test_statistic = my_logrank_test(Z_sim.ravel(), T_hat_sim, C_hat_sim, Delta_hat_sim)#my_logrank_test(Z_sim.ravel(), T_hat_sim, C_hat_sim, Delta_hat_sim)
        test_statistics.append(test_statistic)

        #if verbose and l % 1000 == 0:
            #print(f"Completed {l}/{L} iterations.")

    # Step 3: Calculate p-value
    test_statistics = np.array(test_statistics)
    p_value = np.mean(test_statistics >= a)

    if verbose:
        print(f"Mean of simulated test statistics: {np.mean(test_statistics)}")
        print(f"p-value: {p_value}")

    return p_value
