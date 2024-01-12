import math
import numpy as np
import pandas as pd
from scipy import stats


# synthetic data for an AB test
np.random.seed(42)
A = np.array([np.random.normal(loc=32.80, scale=5) for _ in range(3000)])
B = np.array([np.random.normal(loc=33.90, scale=5) for _ in range(3100)])

print(np.mean(A))
print(np.std(A))
"""
32.960004179379176
4.933215506730164
"""
print(np.mean(B))
print(np.std(B))
"""
33.708782570856776
5.04329023307
"""




# calculate mean ans std of samples
def parameters(A, B):

    x_bar1 = np.mean(A)
    x_bar2 = np.mean(B)
    s1 = np.std(A, ddof=1)  # ddof=1 ensures that the sample std be calculated
    s2 = np.std(B, ddof=1)

    return x_bar1, x_bar2, s1, s2


# calculate degrees of freedom
def freedom_degree(A, B):
    _, _, s1, s2 = parameters(A, B)

    numerator = (s1**2 / len(A) + s2**2 / len(B))**2
    denominator = (
        ((s1**2 / len(A))**2 / (len(A) - 1)) / (len(A) - 1) +
        ((s2**2 / len(B))**2 / (len(B) - 1)) / (len(B) - 1)
    )

    dof = numerator / denominator

    return dof


print(freedom_degree(A, B))
"""18584100.568456642"""


# calculate t-statistic
def t_statistic(A, B):

    xbar1, xbar2, s1, s2 = parameters(A, B)

    t_stats = (xbar1 - xbar2) / np.sqrt((s1 ** 2 / len(A)) + (s2 ** 2 / len(B)))

    return t_stats



"""
With the ability to calculate the t-statistic now you need a way to determine 
if you should reject (or not) the null hypothesis. Complete the h_reject() function below. 
This function should return whether to reject (or not) the null hypothesis by using the p-value 
method given the value of the observed statistic, the degrees of freedom and a level of significance. 
This should be a two-sided test.
"""


def h_reject(A, B, alpha=0.5):

    t_stats = t_statistic(A, B)
    dof = freedom_degree(A, B)

    reject = False
    p_value = 2 * (1 - stats.t.cdf(abs(t_stats), dof))  # Two-sided test

    if p_value < alpha:
        reject = True

    return reject



#####################




@dataclass
class estimation_metrics_prop:
    n: int
    x: int
    p: float

    def __repr__(self):
        return f"sample_params(n={self.n}, x={self.x}, p={self.p:.3f})"


def compute_proportion_metrics(data):
    """Computes the relevant metrics out of a sample for proportion-like data.

    Args:
        data (pandas.core.series.Series): The sample data. In this case 1 if the user converted and 0 otherwise.

    Returns:
        estimation_metrics_prop: The metrics saved in a dataclass instance.
    """

    ### START CODE HERE ###
    metrics = estimation_metrics_prop(
        n=len(data),
        x=data.sum(),  # Sum of 1s in the data gives the number of users who converted
        p=data.mean(),  # Proportion is the mean of the data
    )
    ### END CODE HERE ###

    return metrics


def pooled_proportion(control_metrics, variation_metrics):
    """Compute the pooled proportion for the two samples.

    Args:
        control_metrics (estimation_metrics_prop): The metrics for the control sample.
        variation_metrics (estimation_metrics_prop): The metrics for the variation sample.

    Returns:
        numpy.float: The pooled proportion.
    """

    ### START CODE HERE ###

    x1, n1 = control_metrics.x, control_metrics.n
    x2, n2 = variation_metrics.x, variation_metrics.n

    pp = (x1 + x2) / (n1 + n2)

    ### END CODE HERE ###

    return pp


def z_statistic_diff_proportions(control_metrics, variation_metrics):
    """Compute the z-statistic for the difference of two proportions.

    Args:
        control_metrics (estimation_metrics_prop): The metrics for the control sample.
        variation_metrics (estimation_metrics_prop): The metrics for the variation sample.

    Returns:
        numpy.float: The z-statistic.
    """

    ### START CODE HERE ###

    # Calculate the pooled proportion
    pp = pooled_proportion(control_metrics, variation_metrics)

    # Extract values for control sample
    n1, p1 = control_metrics.n, control_metrics.p

    # Extract values for variation sample
    n2, p2 = variation_metrics.n, variation_metrics.p

    # Calculate the z-statistic
    z = (p1 - p2) / ((pp * (1 - pp) * ((1 / n1) + (1 / n2))) ** 0.5)

    ### END CODE HERE ###

    return z


def reject_nh_z_statistic(z_statistic, alpha=0.05):
    """Decide whether to reject (or not) the null hypothesis of the z-test.

    Args:
        z_statistic (numpy.float): The computed value of the z-statistic for the two proportions.
        alpha (float, optional): The desired level of significancy. Defaults to 0.05.

    Returns:
        bool: True if the null hypothesis should be rejected. False otherwise.
    """

    reject = False
    ### START CODE HERE ###
    # Calculate the two-sided p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))

    if p_value < alpha:
        reject = True
    ### END CODE HERE ###

    return reject


def confidence_interval_proportion(metrics, alpha=0.05):
    """Compute the confidende interval for a proportion-like sample.

    Args:
        metrics (estimation_metrics_prop): The metrics for the sample.
        alpha (float, optional): The desired level of significance. Defaults to 0.05.

    Returns:
        (numpy.float, numpy.float): The lower and upper bounds of the confidence interval.
    """

    ### START CODE HERE ###
    n, p = metrics.n, metrics.x / metrics.n

    # Calculate the critical z-value
    z_value = stats.norm.ppf(1 - alpha / 2)

    # Calculate the margin of error
    distance = z_value * ((p * (1 - p)) / n) ** 0.5

    # Calculate the confidence interval bounds
    lower = p - distance
    upper = p + distance

    return lower, upper
















