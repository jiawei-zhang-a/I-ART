import numpy as np
from lifelines.statistics import logrank_test,multivariate_logrank_test

def kaplan_meier_weight(t_l, distinct_times, D, N):
    """
    Calculate the Kaplan-Meier weight for the Prentice-Wilcoxon test.
    """
    return 1  # Weight is fixed to 1 for equivalence to log-rank test

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

def _my_logrank_test(Z, T, C, Delta):
    """
    Perform a log-rank test between treatment groups using lifelines.
    """
    Z = np.asarray(Z).ravel()  # Ensure Z is a flat 1D array
    T = np.asarray(T)
    C = np.asarray(C)

    observed_times = np.minimum(T, C)

    group1_times = observed_times[Z == 1]
    group1_events = Delta[Z == 1]
    group2_times = observed_times[Z == 0]
    group2_events = Delta[Z == 0]

    results = logrank_test(
        durations_A=group1_times, durations_B=group2_times,
        event_observed_A=group1_events, event_observed_B=group2_events,
        weightings='wilcoxon'
    )

    return results.test_statistic

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

# Equivalence Check
def equivalence_check():
    np.random.seed(42)
    
    # Generate example data
    N = 100
    Z = np.random.choice([0, 1], size=N)  # Treatment group assignment
    T = np.random.uniform(5, 15, size=N)  # Event times
    C = np.random.uniform(10, 20, size=N)  # Censoring times
    Delta = (T <= C).astype(int)  # Event indicator (1 = event, 0 = censored)

    # Compute test statistics
    wilcoxon_stat = wilcoxon_prentice(Z, T, C, Delta)
    logrank_stat = my_logrank_test(Z, T, C, Delta)

    # Print and compare
    print(f"Wilcoxon-Prentice Test Statistic (W(t)=1): {wilcoxon_stat}")
    print(f"Log-Rank Test Statistic (lifelines): {logrank_stat}")
    print(f"Difference: {np.abs(wilcoxon_stat - logrank_stat)}")

# Run equivalence check
equivalence_check()
