from pathlib import Path


from pathlib import Path
from typing import Dict
from statistics import mean, median, mode, stdev

import pandas as pd
from sqlalchemy import false


def root() -> Path:
    return Path(__file__).parent.absolute()


def stats(sr: pd.Series) -> Dict:
    data_lst = sr.tolist()
    
    min_ = min(data_lst)
    max_ = max(data_lst)
    
    mean_ = round(mean(data_lst))
    std_dev = round(stdev(data_lst))
    two_stdev = round(2 * std_dev)
    three_stdev = round(3 * std_dev)
    
    empirical_rule_68 = (round(mean_ - std_dev), round(mean_ + std_dev))
    empirical_rule_95 = (round(mean_ - two_stdev), round(mean_ + two_stdev))
    empirical_rule_99 = (round(mean_ - three_stdev), round(mean_ + three_stdev))
    
    empirical_rule = [empirical_rule_68, empirical_rule_95, empirical_rule_99]
    
    empirical_rule_is_out_of_bounds = False
    for rule in empirical_rule:
        if rule[0] < min_ or rule[1] > max_:
            empirical_rule_is_out_of_bounds = True
    
    dict = {
        'min': min(data_lst),
        'max': max(data_lst),
        'mean': mean_,
        'median': round(median(data_lst)),
        'mode': mode(data_lst),
        'stdev': std_dev,
        '2_stdev': two_stdev,
        '3_stdev': three_stdev,
        'empirical_rule_68': empirical_rule_68,
        'empirical_rule_95': empirical_rule_95,
        'empirical_rule_99': empirical_rule_99, 
        'empirical_rule_is_out_of_bounds': empirical_rule_is_out_of_bounds,
        'coefficient_of_variation': round(std_dev / mean_, 3)
    }
    
    return dict