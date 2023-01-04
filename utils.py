from pathlib import Path

import statsmodels.stats.weightstats as ws
import matplotlib.pyplot as plt
from random import choice
from pathlib import Path
from math import pow, sqrt, asin
from typing import Dict, List, Tuple, Callable
from statistics import mean, median, mode, stdev
from statsmodels.stats.proportion import proportions_ztest

import pandas as pd
from sqlalchemy import false


def root() -> Path:
    return Path(__file__).parent.absolute()


def median_(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(median).map(_round)


def run_continuous_test(empl_samples: List, attr_samples: List, colname: str) -> pd.Series:
    num_of_samples = len(empl_samples)
    lst = []
    for x in range(num_of_samples):
        empl_sample = empl_samples[x]
        attr_sample = attr_samples[x]

        df = _continuous_test(empl_sample, attr_sample, colname)
        lst.append(df)

    results = pd.concat(lst, ignore_index=True)
    return results


def average(empl_pop: pd.DataFrame, attr_pop: pd.DataFrame, colname: str) -> Dict:
    empl_grp = empl_pop[colname]
    attr_grp = attr_pop[colname]
    
    empl_mean = int(mean(empl_grp))
    attr_mean = int(mean(attr_grp))
    
    return {'empl': empl_mean, 'attr': attr_mean}
    
    
def pie_chart(empl_pop: pd.DataFrame, attr_pop: pd.DataFrame, colname: str) -> None:
    fig = plt.figure(figsize=(4,3),dpi=144)
    
    ax1 = fig.add_subplot(121)
    empl_counts = empl_pop[colname].value_counts().sort_index().tolist()
    print(empl_pop[colname].value_counts().sort_index())
    labels = ['0', '1', '2', '3']
    empl_colors = ['r', 'b']
    ax1.pie(empl_counts, labels=labels, colors=empl_colors, autopct='%1.2f%%', startangle = 90)

    ax2 = fig.add_subplot(122)
    attr_counts = attr_pop[colname].value_counts().sort_index().tolist()
    print(attr_pop[colname].value_counts().sort_index())
    ax2.pie(attr_counts, labels=labels, startangle = 90)

    plt.tight_layout()
    plt.show()


def box_plot(empl_pop: pd.DataFrame, attr_pop: pd.DataFrame, colname: str) -> None:
    data = [empl_pop[colname], attr_pop[colname]]
    plt.boxplot(data)
    plt.xticks([1, 2], ['Empl', 'Attr'])
    plt.title(f'{colname}')
    
    print('\n')
    print(_title('box plot'))
    plt.show()
    
    
def bar_chart_avg(empl_pop: pd.DataFrame, attr_pop: pd.DataFrame, colname: str) -> None:
    avgs = average(empl_pop, attr_pop, colname)
    
    y = ['Employed', 'Attrited']
    x = [avgs['empl'], avgs['attr']]
    
    fig, ax = plt.subplots()
    bars = ax.barh(y, x)

    ax.bar_label(bars)\
        
    colors = _palette()
    bars[0].set_color(colors[0])
    bars[1].set_color(colors[1])
    
    plt.xlabel(colname)
    plt.title(f'Average {colname}')
    
    print('\n')
    print(_title('bar chart'))
    plt.show()
    

def bar_chart_shift(empl_pop: pd.DataFrame, attr_pop: pd.DataFrame, colname: str) -> None:
    empl_shift_count = empl_pop[colname].value_counts()[0]
    attr_shift_count = attr_pop[colname].value_counts()[0]
    
    y = ['Employed', 'Attrited']
    x = [round(empl_shift_count / empl_pop.shape[0], 2) * 100, 
         round(attr_shift_count / attr_pop.shape[0], 2) * 100]
    
    fig, ax = plt.subplots()
    bars = ax.barh(y, x)

    ax.bar_label(bars)
        
    colors = _palette()
    bars[0].set_color(colors[0])
    bars[1].set_color(colors[1])
    
    plt.xlabel('Percentage')
    plt.title(f'Percentage of Employees on Split Shift')
    
    print('\n')
    print(_title('bar chart'))
    plt.show()
    
    
def bar_chart_ot(empl_pop: pd.DataFrame, attr_pop: pd.DataFrame, colname: str) -> None:
    attr_prop = attr_pop[colname].value_counts()['Yes'] / attr_pop[colname].shape[0]
    empl_prop = empl_pop[colname].value_counts()['Yes'] / empl_pop[colname].shape[0]
    
    attr_prop = round(attr_prop, 2) * 100
    empl_prop = round(empl_prop, 2) * 100
    
    y = ['Employed', 'Attrited']
    x = [empl_prop, attr_prop]
    
    fig, ax = plt.subplots()
    bars = ax.barh(y, x)

    ax.bar_label(bars)
    colors = _palette()
    bars[0].set_color(colors[0])
    bars[1].set_color(colors[1])
    
    plt.xlabel('Percentage')
    plt.title(f'Percentage of Employees w/ Over Time')
    
    print(_title('bar chart'))
    plt.show()
    
    
def _palette() -> List:
    palette1 = ['#468189', '#F4E9CD']
    palette2 = ['#6C9A8B', '#E8998D']
    palette3 = ['#1A5E63', '#F0F3BD']
    palette4 = ['#084887', '#F9AB55']
    palette5 = ['#62929E', '#C6C5B9']
    
    lst = [palette1, palette2, palette3, palette4, palette5]
    return choice(lst)


def run_test(empl_samples: List, attr_samples: List, *, func: Callable=None, colname: str='') -> pd.Series:
    
    num_of_samples = len(empl_samples)
    lst = []
    for x in range(num_of_samples):
        empl_sample = empl_samples[x]
        attr_sample = attr_samples[x]
        
        df = None
        if func != None:
            df = func(empl_sample, attr_sample)
        else:
            df = _quant_test(empl_sample, attr_sample, colname)
        lst.append(df)
    
    return pd.concat(lst, ignore_index=True)


def cohen_h(p1: float, p2: float) -> float:
    p1_sqrt, p2_sqrt = sqrt(p1), sqrt(p2)
    h = (asin(p1_sqrt) - asin(p2_sqrt)) * 2
    return round(abs(h), 4)


def diff_between_means(empl_sample: pd.Series, attr_sample: pd.Series) -> int:
    x_bar = mean(empl_sample)
    x_bar2 = mean(attr_sample)
    return abs(x_bar - x_bar2)


def proportions_pval(x1: int, n1: int, x2: int, n2: int) -> float:
    count = [x1, x2]
    nobs = [n1, n2]
    return proportions_ztest(count=count, nobs=nobs)[1]


def prop_pval_cohens_h(empl_sample: pd.DataFrame, attr_sample: pd.DataFrame, colname: str) -> float:
    x1 = empl_sample[colname].value_counts()['Yes']
    x2 = attr_sample[colname].value_counts()['Yes']
    n = empl_sample.shape[0]
    
    count = [x1, x2]
    nobs = [n, n]
    return proportions_ztest(count=count, nobs=nobs)[1]


def _title(charttype: str) -> str:
    dash_length = 15
    title = charttype.upper()
    title = f'\n{title}\n'
    for _ in range(dash_length):
        title += '-'
    return title


def _ci(empl_grp: pd.Series, attr_grp: pd.Series) -> Tuple:
    '''
    Uses the empirical rule to calculate the confidence interval (CI) for the group w/ the lowest mean. A note about using the CI w/ the lowest mean: choosing the lowest mean enables efficient comparison. We could have chosen to use the highest mean (or both), but using the lowest aligns w/ general thought that low pay is a major factor in attrition (we acknowledge that there are many factors), and since attrition is the focus of this project, we chose to use the lowest. 
    
    Returned tuple includes proportions for each group where observations fall within the CI.
        
    @param:
        empl_grp (pd.Series): group of employed employees
        attr_grp (pd.Series): group of attrited employees

    @returns:
        Tuple: (CI, empl_proportion, attr_proportion)
    '''
    
    empl_stats = _stats(empl_grp)
    attr_stats = _stats(attr_grp)
    
    ci = None
    if empl_stats['mean'] <= attr_stats['mean']:
        ci = empl_stats['ci_68']
    else:
        ci = attr_stats['ci_68']
    
    empl_p = _proportion(empl_grp, ci)
    attr_p = _proportion(attr_grp, ci)
    return (ci, empl_p, attr_p)


def cohens_d(empl_sample: pd.Series, attr_sample: pd.Series, colname: str) -> float:
    empl_sample = empl_sample[colname]
    attr_sample = attr_sample[colname]
    
    x_bar = mean(empl_sample)
    x_bar2 = mean(attr_sample)
    
    stddev = stdev(empl_sample)
    stddev2 = stdev(attr_sample)
    
    
    def _pooled_stdev(stdev: float, stdev2: float) -> float:
        _ = (pow(stdev, 2) + pow(stdev2, 2)) / 2
        return sqrt(_)

    
    d = (x_bar2 - x_bar) / _pooled_stdev(stddev, stddev2)
    return abs(round(d, 4))


def _means(empl_grp: pd.Series, attr_grp: pd.Series) -> Tuple:
    '''
    Calculates mean for each group. Rounded to two decimal places.
        
    @param:
        empl_grp (pd.Series): group of employed employees
        attr_grp (pd.Series): group of attrited employees

    @returns:
        Tuple: (empl_mean, attr_mean)
    '''
    empl_mean = int(mean(empl_grp))
    attr_mean = int(mean(attr_grp))
    return (empl_mean, attr_mean)


def _quant_test(empl_sample: pd.DataFrame, attr_sample: pd.DataFrame, colname: str) -> pd.DataFrame:
    empl_grp = empl_sample[colname]
    attr_grp = attr_sample[colname]
    
    pval = _pval_unequal_stdev(empl_grp, attr_grp)
    cohens_d = _cohens_d(empl_grp, attr_grp)
    
    ci, empl_p, attr_p = _ci(empl_grp, attr_grp)
    empl_mean, attr_mean = _means(empl_grp, attr_grp)
    
    dict_ = {
        'pval': [pval],
        'cohens_d': [cohens_d],
        'ci_68_lower': [ci[0]],
        'ci_68_upper': [ci[1]],
        'empl_p_within_ci': [empl_p],
        'attr_p_within_ci': [attr_p],
        'empl_mean': [empl_mean],
        'attr_mean': [attr_mean]
    }
    return pd.DataFrame(dict_)


def _round(x):
    return round(x, 3)


def _proportion(group: pd.Series, ci: Tuple) -> float:
    '''
    Determines the proportion of a group's observations that fall within the confidence interval.
        
    @param:
        group (pd.Series): group of observations
        ci (Tuple): confidence interval; the 1st element is the lower ci, the 2nd is the upper

    @returns:
        float: proportion (rounded to two decimals)
    '''
    n = group.shape[0]
    obs = 0
    
    for x in group:
        if x >= ci[0] and x <= ci[1]:
            obs += 1
    
    p = round((obs / n), 2)
    return p


def pval_unequal_stdev(empl_sample: pd.Series, attr_sample: pd.Series, colname: str) -> float:
    sample1 = ws.DescrStatsW(empl_sample[colname])
    sample2 = ws.DescrStatsW(attr_sample[colname])
    
    cm_obj = ws.CompareMeans(sample1, sample2)
    
    zstat, pval = cm_obj.ztest_ind(usevar='unequal')
    return round(pval, 4)


def print_significance(pval: float, cohens: float) -> None:
    pval_sig = ''
    cohen_effect = ''
    
    if pval <= 0.01:
        pval_sig = 'High Significance'
    elif pval > 0.01 and pval <= 0.05:
        pval_sig = 'Significance'
    else:
        pval_sig = 'Nonsignificance'
        
    if cohens < 0.2:
        cohen_effect = 'Zero'
    elif cohens >= 0.2 and cohens <= 0.4:
        cohen_effect = 'Small'
    elif cohens > 0.4 and cohens <= 0.7:
        cohen_effect = 'Medium'
    else:
        cohen_effect = 'Large'
    
    if pval < 0.01:
        pval = '< 0.01'
    
    title = '\nSIGNIFICANCE\n---------------'
    print(f'\n{title}\nP value of {pval} shows {pval_sig} and Cohens of {cohens} shows a {cohen_effect} effect size.\n')


def _stats(group: pd.Series) -> Dict:
    stdev_ = stdev(group)
    mean_ = mean(group)
    min_ = min(group)
    
    lower = mean_ - stdev_
    upper = mean_ + stdev_
    
    lower = min_ if lower < min_ else lower
    dict_ = {
        'stdev': round(stdev_),
        'mean': round(mean_),
        'ci_68': (round(lower), round(upper))
    }
    return dict_