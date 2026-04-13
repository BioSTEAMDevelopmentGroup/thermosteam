# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
import inspect
import os
import sys
import tokenize
import linecache

__all__ = (
    'infer_variable_assignment',
)

cache = {}
max_filecache = 50
def infer_variable_assignment(caller):
    # Get frame where item is assigned. 
    # Stacklevel default to 2, assuming this method is used within `__init__`
    frame = inspect.currentframe().f_back.f_back
    filename = frame.f_code.co_filename
    
    # This means no variable assignment.
    # Registered BioSTEAM objects will know how to handle this.
    default = '-'
    
    # Get current line from frame file
    try:
        lines, variable_cache, crossout = cache[filename]
    except:
        # If the file fails to load, lines will be None and this is cached.
        lines = None
        # If using native cache, we replace the lines that failed with None
        # to save space and improve speed.
        crossout = True
        if filename in linecache.cache:
            try: 
                # Must be ipython or notebook session
                lines = linecache.cache[filename][2]
                crossout = False
            except: 
                lines = None
        if lines is None:
            if os.path.isabs(filename):
                fullname = filename
            else:
                # Try looking through the module search path, which is only useful
                # when handling a relative filename.
                for dirname in sys.path:
                    try: fullname = os.path.join(dirname, filename)
                    except (TypeError, AttributeError): continue
                    else: break
            try:
                with tokenize.open(fullname) as fp: lines = fp.readlines()
            except:
                pass
        # We use variable_cache to cache the lines that we already tried 
        # and not attempt them again; this improves speed.
        variable_cache = None if lines is None else {}
        cache[filename] = lines, variable_cache, crossout
        if cache.__len__() > max_filecache:
            del cache[cache.__iter__().__next__()]
    if lines is None: return default
    n = frame.f_lineno - 1
    if n in variable_cache:
        return variable_cache[n]
    else:
        line = lines[n]
    
    # Attempt to infer name
    try: 
        line = line.strip()
        if line.startswith('with '): # Context management
            callside, nameside = line.split(' as ', 1)
            callside = callside[5:].strip() # remove 'with ' and trailing spaces
            variable_name = nameside.split(':', 1)[0].strip() # remove ':' and trailing spaces
            valid_variable_name = variable_name.isidentifier()
        else: # Ordinary variable assignment
            nameside, callside = line.split('=', 1)
            names = nameside.strip().split('.')
            variable_name = names[-1]
            valid_variable_name = all([i.isidentifier() for i in names])
        if valid_variable_name:
            # Make sure caller is correct (e.g., not a nested call)
            callside = callside.strip()
            name = callside.split('(', 1)[0]
            key, *other = name.split('.')
            if key in frame.f_locals:
                obj = frame.f_locals[key]
            elif key in frame.f_globals:
                obj = frame.f_globals[key]
            else:
                if crossout: lines[n] = None
                variable_cache[n] = default
                return default # NameError by user
            for key in other: obj = getattr(obj, key)
            if obj != caller: variable_name = default # No assignment
        else:
            variable_name = default # No assignment
        variable_cache[n] = variable_name
        return variable_name
    except:
        if crossout: lines[n] = None
        variable_cache[n] = default
        return default # Could not infer name