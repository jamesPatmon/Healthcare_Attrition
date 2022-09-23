from pathlib import Path

import statsmodels.stats.weightstats as ws
from pathlib import Path
from math import pow, sqrt, asin
from typing import Dict, List, Tuple, Callable
from statistics import mean, median, mode, stdev
from statsmodels.stats.proportion import proportions_ztest

import pandas as pd
from sqlalchemy import false


def root() -> Path:
    return Path(__file__).parent.absolute()


def run_test(func: Callable, empl_samples: List, attr_samples: List, is_all_quant: bool=False) -> pd.Series:
    num_of_samples = len(empl_samples)
    lst = []
    for x in range(num_of_samples):
        empl_sample = empl_samples[x]
        attr_sample = attr_samples[x]
        
        df = func(empl_sample, attr_sample)
        lst.append(df)

    results = pd.concat(lst, ignore_index=True)
    
    if is_all_quant:  
        median_results = results.apply(median).map(_round)
        return median_results
    else:
        return results


def cohen_h(p1: float, p2: float) -> float:
    p1_sqrt, p2_sqrt = sqrt(p1), sqrt(p2)
    h = (asin(p1_sqrt) - asin(p2_sqrt)) * 2
    return round(abs(h), 4)


def cohens_d(empl_sample: pd.Series, attr_sample: pd.Series) -> float:
    x_bar = mean(empl_sample)
    x_bar2 = mean(attr_sample)
    
    stddev = stdev(empl_sample)
    stddev2 = stdev(attr_sample)
    
    
    def pooled_stdev(stdev: float, stdev2: float) -> float:
        _ = (pow(stdev, 2) + pow(stdev2, 2)) / 2
        return sqrt(_)

    
    d = (x_bar2 - x_bar) / pooled_stdev(stddev, stddev2)
    return abs(d)


def diff_between_means(empl_sample: pd.Series, attr_sample: pd.Series) -> int:
    x_bar = mean(empl_sample)
    x_bar2 = mean(attr_sample)
    return abs(x_bar - x_bar2)


def means(empl_sample: pd.Series, attr_sample: pd.Series) -> Tuple:
    '''
    Calculates mean for each sample. Rounded to two decimal places.
        
    @param:
        empl_sample (pd.Series): sample of employed employees
        attr_sample (pd.Series): sample of attrited employees

    @returns:
        Tuple: (empl_mean, attr_mean)
    '''
    empl_mean = int(mean(empl_sample))
    attr_mean = int(mean(attr_sample))
    return (empl_mean, attr_mean)


def ci(empl_sample: pd.Series, attr_sample: pd.Series) -> Tuple:
    '''
    Uses the empirical rule to calculate the confidence interval (CI) for the sample w/ the lowest mean. 
    Returned tuple includes proportions for each sample where observations fall within the CI.
        
    @param:
        empl_sample (pd.Series): sample of employed employees
        attr_sample (pd.Series): sample of attrited employees

    @returns:
        Tuple: (CI, empl_proportion, attr_proportion)
    '''
    empl_stats = _stats(empl_sample)
    attr_stats = _stats(attr_sample)
    
    ci = None
    if empl_stats['mean'] <= attr_stats['mean']:
        ci = empl_stats['ci_68']
    else:
        ci = attr_stats['ci_68']
    
    empl_p = _proportion(empl_sample, ci)
    attr_p = _proportion(attr_sample, ci)
    return (ci, empl_p, attr_p)


def pval_unequal_stdev(empl_sample: pd.Series, attr_sample: pd.Series) -> float:
    sample1 = ws.DescrStatsW(empl_sample)
    sample2 = ws.DescrStatsW(attr_sample)
    
    cm_obj = ws.CompareMeans(sample1, sample2)
    
    zstat, pval = cm_obj.ztest_ind(usevar='unequal')
    return pval


def proportions_pval(x1: int, n1: int, x2: int, n2: int) -> float:
    count = [x1, x2]
    nobs = [n1, n2]
    return proportions_ztest(count=count, nobs=nobs)[1]
    
    
def _round(x):
    return round(x, 3)


def _proportion(sample: pd.Series, ci: Tuple) -> float:
    n = sample.shape[0]
    obs = 0
    
    for x in sample:
        if x >= ci[0] and x <= ci[1]:
            obs += 1
    
    p = round((obs / n), 2)
    return p


def _stats(sample: pd.Series) -> Dict:
    stdev_ = stdev(sample)
    mean_ = mean(sample)
    
    lower = mean_ - stdev_
    upper = mean_ + stdev_
    
    dict_ = {
        'stdev': round(stdev_),
        'mean': round(mean_),
        'ci_68': (round(lower), round(upper))
    }
    return dict_