# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2021-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

def longest_common_prefix(string_list):
    """
    Find a common prefix between a list of strings (e.g. filenames)
    :param string_list: a list of strings for which we will search the common prefix,
        i.e. the longest first characters which belong to all strings
    :return: the string with the common prefix
    """
    c = ""
    break_ = False
    for i in range(1, len(string_list[0]) + 1):
        c = string_list[0][:i]
        for s in string_list[1:]:
            if len(s) < i + 1:
                break_ = True
                break
            if s[:i] != c:
                break_ = True
                break
        if break_:
            break
    return c
