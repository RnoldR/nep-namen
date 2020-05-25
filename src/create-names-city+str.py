#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:19:05 2019

@author: arnold
"""

import os
import re
import sys
import yaml
import mypy
import types
import random
import pandas as pd # type: ignore
import matplotlib.pyplot as plt

import csv_to_table as cvt

from typing import List, Set, Dict, Tuple, Optional, Any

import logging
import importlib
importlib.reload(logging)


def import_namen(creds: str, server: str, port: str, database: str,
                 tabel_namen: List, col_namen: List, where: str=None) -> pd.Series:
    """
    Import a set of column names from a Postgres table.

    Args:
        server (str): Name of server.
        port (str): Port of postgres server.
        database (str): Name of the database.
        tabel_namen (List): List of tables to import from.
        col_namen (List): list of column names, one column for each table.
        where (str, optional): optional Sql WHERE clase to select over the columns.

    Returns:
        namen (pandas.Series): A series of names.

    """
    # Haal namen op uit de database
    result = None
    for tabel_naam, col_naam in zip(tabel_namen, col_namen):
        namen = converter.read_table(creds, server, database,
                                     tabel_naam, port=port, columns=col_naam,
                                     where=where)
        if result is None:
            result = pd.Series(namen[col_naam])
        else:
            result = result.append(pd.Series(namen[col_naam]), ignore_index=True)

    # remove all multiple occurences
    namen = pd.Series(list(set(result)))

    # lowercase de boel en verwijder 'wijk <nn> ' aan het begin
    namen = namen.str.lower().str.replace('^wijk [0-9]*', '').str.strip()

    # Verwijder e.o. aan het eind
    namen = namen.str.replace('e.o.', '').str.strip()

    return namen

## import_namen ##


def load_cache(cache_name: str) -> pd.Series:
    """
    Load data from cache

    Args:
        cache_name (str): File name of the cache.

    Returns:
        namen (pandas.Series): A series of names.

    """
    namen = []
    with open(cache_name, 'r') as cache:
        for line in cache:
            namen.append(line.strip())

    return pd.Series(namen)

## load_cache ##


def get_data(creds, server: str, port: str, database: str,
             tabel_namen: List, col_namen: List, where: str=None,
             cache_name=None) -> pd.Series:
    """
    Fetch data, either from a postgres database when cache_name is none, else
    from the cache.

    Args:
        server (str): Name of server.
        port (str): Port of postgres server.
        database (str): Name of the database.
        tabel_namen (List): List of tables to import from.
        col_namen (List): list of column names, one column for each table.
        where (TYPE, optional): optional Sql WHERE clase to select over the columns.
        cache_name (str, optional): File name of the cache. Defaults to None.

    Returns:
        namen (pandas.Series): A series of names.

    """
    if cache_name is not None and os.path.exists(cache_name):
        namen = load_cache(cache_name)
        print('Loaded from cache', cache_name)
    else:
        namen = import_namen(creds, server, port, database,
                             tabel_namen, col_namen, where=where)

        if cache_name is not None:
            with open(cache_name, 'w') as cache:
                namen_to_save = list(set(namen))
                for naam in namen_to_save:
                    print(naam, file=cache)
                # for
            # with
        # if
    # if

    return namen

## get_data ##


def clean_namen(namen: pd.Series) -> pd.Series:
    """
    For each row in namen, break it up in separate words and create a new
    series out of it. Remove names with length 2 or shorter

    Args:
        namen (pandas.Series): Series of names.

    Returns:
        namen (pandas.Series): Series of cleaned names.

    """
    # Splits naam op basis van non-alphanumerics
    namen_lijst = []
    for naam in namen:
        stukken = re.findall(r"[^\W]+", naam)
        if len(stukken) > 0:
            namen_lijst.extend(stukken)

    # Verwijder onzin namen
    onzin = ['de', 'den', 'het', 'een', 'ee', 'ter', '½n', '+', 'ó']
    namen = list(set(filter(lambda w: w not in onzin, namen_lijst)))

    # Verwijder verdwenen of te korte namen (met lengte < 3)
    namen = pd.Series(namen)
    namen = namen[namen.map(len) > 2]

    return namen

## clean_namen ##


def deconstruct_namen(namen: pd.Series, prefixes: pd.Series, suffixes: pd.Series):
    """
    Convert all names to their stems, meaning without their pre- and suffixes.

    Args:
        namen (pd.Series): Series of names.
        prefixes (pd.Series): Series of prefixes.
        suffixes (pd.Series): Series of suffixes.

    Returns:
        namen (pd.Series): Series of names.
        pre_count (dict): dict of occurrences of prefixes.
        suf_count (dict): dict of occurrences of suffixes.

    """

    # Count the occurrence of pre- and suffixes for later use in probability distributions
    pre_count = {x: sum(namen.str.count('^' + x)) for x in list(prefixes)}
    suf_count = {x: sum(namen.str.count(x + '$')) for x in suffixes}
    pre_sum = sum(pre_count.values())
    suf_sum = sum(suf_count.values())

    # Remove pre- and suffixes. A removal can only occur once. Multiple removing
    # causes problems with Dutch names where lots of names can be constructed from
    # several pre- and suffixes. So first an occurrence is replaced by '*',
    # so a pre-/suffix will not be found anymore, next all '*' are replaced
    # by emptyness.
    for prefix in prefixes:
        namen = namen.str.replace('^' + prefix, '*')
    for suffix in suffixes:
        namen = namen.str.replace(suffix + '$', '*')

    # Replace '*' by empty string and remove names with length < 3
    namen = namen.str.replace('\\*', '')
    namen = list(set(namen[namen.map(len) > 2]))

    pre_count[''] = len(namen) - pre_sum
    suf_count[''] = len(namen) - suf_sum

    return namen, pre_count, suf_count

## deconstruct_namen ##


def reconstruct_namen(stems: pd.Series, existing: pd.Series,
                      pre_count: dict, suf_count: dict, n: int):
    """
    Create new names based on a Series of stems and pre- and suffixes.
    The pre- and suffixes are dictionaries with their weights (in this
    case occurrences from the counts in deconstruct_names) which
    will be used as weights in random.choices.
    The newly created names are tested for their existence, if it already
    exists the new name is discarded.

    Args:
        stems (pd.Series): Series of stems.
        existing (pd.Series): Series of existing names.
        pre_count (dict): dictionary of prefixes and weights.
        suf_count (dict): dictionary of suffixes and weights.
        n (int): Number of names to be generated.

    Returns:
        namen (list): List of newly created names.

    """

    prefixes = list(pre_count.keys())
    pre_weights = list(pre_count.values())
    suffixes = list(suf_count.keys())
    suf_weights = list(suf_count.values())

    namen = []
    counted = 0
    while counted < n:
        random_stem = random.randint(0, len(stems) - 1)
        stem = stems[random_stem]
        prefix = random.choices(prefixes, pre_weights)[0]
        suffix = random.choices(suffixes, suf_weights)[0]

        # add new name to namen when it cannot be found in the list
        # of existing names
        result = (prefix + ' ' + stem + suffix).strip()
        existing_name_list = list(existing)
        if result not in existing_name_list:
            counted += 1
            namen.append(result)
        # if
    # while

    return namen

## reconmstruct_namen ##


def generate_namen(creds: str, server: str, port: str, database: str,
                   prefixes, suffixes, namenfile: str,
                   tabel_namen: list, col_namen: list, n: int,
                   cache_name: str=None, where=None):

    namen = get_data(creds, server, port, database, tabel_namen, col_namen,
                     where=where, cache_name=cache_name)

    print(len(namen), 'names read.')

    stems, pre_count, suf_count = deconstruct_namen(namen, prefixes, suffixes)
    print(pre_count, '\n)')
    print(suf_count, '\n)')
    print(len(stems), 'usable stems.')

    cleanups = clean_namen(stems)
    print(len(cleanups), 'names left after cleanups.')

    new_names = reconstruct_namen(stems, namen, pre_count, suf_count, n)
    with open(namenfile, 'w') as outfile:
        for name in new_names:
            print(name, file=outfile)

    print(new_names[:25])

    return new_names

## generate_namen ##


PG_CREDS: str = '/home/arnold/.pgpass'
CONFIG_PLAATS: str = 'create-names-city+str-plaats.yaml'
CONFIG_STRAAT: str = 'create-names-city+str-straat.yaml'
GENERATE: List = [CONFIG_PLAATS, CONFIG_STRAAT]

if __name__ == '__main__':
    # create logger
    logger = logging.getLogger('csv_to_table_test')

    logger.setLevel(10)

    # create file handler which logs even debug messages
    fh = logging.FileHandler('csv-reader.log')
    fh.setLevel(logging.INFO)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(logging.Formatter('%(message)s'))

    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    # Initialize important parameters
    pd.options.display.max_rows = 999999
    pd.options.display.max_columns = 999999
    random.seed(42)
    converter = cvt.csv_to_table()

    for config_file in GENERATE:
        print('\nProcessing config from:', config_file)
        with open(config_file) as yaml_data:
            hyper_pars = yaml.load(yaml_data, Loader=yaml.FullLoader)

        host = hyper_pars['host']
        port = hyper_pars['port']
        db = hyper_pars['db']
        tables = hyper_pars['tables']
        columns = hyper_pars['columns']
        destination = hyper_pars['destination']
        user = hyper_pars['user']
        n = hyper_pars['n']
        prefixes = hyper_pars['prefixes']
        suffixes = hyper_pars['suffixes']

        if user == '':
            credentials = PG_CREDS
        else:
            credentials = (user, hyper_pars['pw'])

        plaatsnamen = generate_namen(credentials, host, port, db, prefixes, suffixes,
                                     destination, tables, columns, n)
