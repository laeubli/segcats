#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Provides helper functions for file I/O operations.
"""

def float_to_str ( float_nr ):
    """
    Returns a more precise string representation of @param float_nr (float) than str(float_nr).
    @return (str): the string representation of @param float_nr
    """ 
    return format(float_nr, '.32f')