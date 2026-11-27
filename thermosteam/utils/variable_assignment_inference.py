# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
Infer the variable a new object is assigned to (e.g., `feed = Stream()` -> 'feed').

Python runs code as "bytecode": a list of simple instructions, such as
"call this function" or "store the result in variable `feed`". Intermediate
values are kept in a pile called the "stack": instructions take values from
the top and put their results back on top. By following the new object through
the instructions after its creation, we find where it is stored. This works
for statements that span many lines and for unpacking, like
`feed, product = Stream(), Stream()`. If reading the bytecode fails
unexpectedly, we fall back to reading the line of source code instead.
"""
import os
import sys
import dis
import tokenize
import linecache

__all__ = (
    'infer_variable_assignment',
)

# This means no variable assignment.
# Registered BioSTEAM objects will know how to handle this.
default = '-'

# %% Bytecode inference

# Instructions that store values. Variables at the module level are listed
# in `code.co_names`; variables inside functions are found with `_varname_from_oparg`.
MODULE_STORES = frozenset(('STORE_NAME', 'STORE_GLOBAL'))
STORES = frozenset((
    *MODULE_STORES, 'STORE_FAST', 'STORE_DEREF', 'STORE_ATTR', 'STORE_SUBSCR',
    'STORE_FAST_STORE_FAST', 'STORE_FAST_LOAD_FAST', # Python 3.13+; two steps in one
))
# Instructions between the call and the store in `with call() as name`
WITH_STEPS = frozenset(('BEFORE_WITH', 'COPY', 'SWAP', 'LOAD_SPECIAL', 'CALL'))
# Instructions that may skip ahead (e.g., `if` and `for`); we do not follow these
JUMPS = frozenset(dis.hasjrel + dis.hasjabs)
HAS_ARGUMENT = frozenset(dis.hasarg)

# Bytecode is a sequence of 2-byte units: an instruction number and its argument.
# CACHE units are placeholders to skip; EXTENDED_ARG units carry extra 
# digits for the next argument when it does not fit in one byte.
CACHE = dis.opmap['CACHE']
EXTENDED_ARG = dis.opmap['EXTENDED_ARG']
instruction_names = dis.opname

def instructions_after(code, position):
    """
    Yield the name, number, and argument of each instruction after the
    one at `position` (its byte index in the bytecode).
    """
    bytecode = code.co_code
    argument = 0
    for i in range(position + 2, len(bytecode), 2):
        number = bytecode[i]
        if number == CACHE: continue
        argument = argument << 8 | bytecode[i + 1]
        if number == EXTENDED_ARG: continue
        yield instruction_names[number], number, argument
        argument = 0

def assigned_name(code, position):
    """
    Return the variable that stores the result of the call at `position`, 
    or `default` if the result is not stored in a variable.
    """
    instructions = instructions_after(code, position)
    name, number, argument = next(instructions)
    if name == 'BEFORE_WITH' or name == 'COPY': # with call() as name; a = b = call()
        while name in WITH_STEPS: name, number, argument = next(instructions)
        return stored_variable(code, name, argument, 0) or default
    depth = 0 # Number of values above the call result in the stack
    while True:
        if name in STORES:
            variable = stored_variable(code, name, argument, depth)
            if variable is not None: return variable
            depth += dis.stack_effect(number, argument)
        elif name == 'SWAP': # Swaps the top value with the one `argument - 1` below
            if depth == 0: depth = argument - 1
            elif depth == argument - 1: depth = 0
        elif (name == 'BUILD_TUPLE' or name == 'BUILD_LIST') and depth < argument:
            # a, b = (call(), ...): the tuple must be unpacked right away
            size = argument
            index = size - 1 - depth # Position of the result in the tuple
            name, number, argument = next(instructions)
            if name == 'UNPACK_SEQUENCE' and argument == size:
                depth = index # Unpacking puts the first item on top
            elif name == 'UNPACK_EX': # a, *b, c = (call(), ...)
                # From the top: items before the star, the starred list, items after
                before = argument & 255
                first_after = size - (argument >> 8)
                if index < before: depth = index
                elif index >= first_after: depth = before + 1 + index - first_after
                else: return default # Starred items are not inferred
            else:
                return default
        elif number in JUMPS:
            return default
        elif name != 'NOP':
            depth += dis.stack_effect(number, argument if number in HAS_ARGUMENT else None)
            # Values above the result never sink below it unless the
            # result was used (e.g., `f(call())` or `[call() for i in x]`)
            if depth <= 0: return default
        name, number, argument = next(instructions)

def stored_variable(code, name, argument, depth):
    """
    Return the variable that store instruction `name` assigns the value
    at `depth` in the stack to, `default` if it uses the value otherwise
    (e.g., `call()[0] = 1`), or None if it does not use the value.
    """
    if name == 'STORE_ATTR': # obj.attr = value; `obj` is on top of `value`
        if depth == 0: return default
        elif depth > 1: return None
        variable = code.co_names[argument]
    elif name == 'STORE_SUBSCR': # obj[key] = value
        return default if depth <= 2 else None
    elif name == 'STORE_FAST_STORE_FAST': # The top value goes to the first variable
        if depth > 1: return None
        variable = code._varname_from_oparg(argument >> 4 if depth == 0 else argument & 15)
    elif depth > 0:
        return None
    elif name == 'STORE_FAST_LOAD_FAST':
        variable = code._varname_from_oparg(argument >> 4)
    elif name in MODULE_STORES:
        variable = code.co_names[argument]
    else:
        variable = code._varname_from_oparg(argument)
    # Hidden variables (e.g., `@py_assert1` made by pytest) are not valid names
    return variable if variable.isidentifier() else default


class CallSites(dict):
    """
    Remembers the variable assigned at each call in a piece of compiled
    code (`code`), keyed by the call's position in the bytecode.
    Each call is only inferred once.
    """
    __slots__ = ('code',)

    def __init__(self, code):
        # Keeping `code` alive ensures `id(code)` is not reused by another object
        self.code = code


# Only the most recent code objects are remembered, so memory stays small
# (e.g., each run of a notebook cell creates a new code object).
call_sites = {} # id(code) -> CallSites
max_call_sites = 256

def infer_variable_assignment(caller):
    # The frame (running function or module) that creates the object.
    # Stack level 2 assumes this is called within `__init__`.
    frame = sys._getframe(2)
    try:
        # `f_lasti` is the position of the call being run in that frame
        return call_sites[id(frame.f_code)][frame.f_lasti]
    except KeyError: # Call not seen before
        return infer_and_cache(frame, caller)

def infer_and_cache(frame, caller):
    code = frame.f_code
    try:
        sites = call_sites[id(code)]
    except KeyError:
        sites = call_sites[id(code)] = CallSites(code)
        if len(call_sites) > max_call_sites: del call_sites[next(iter(call_sites))]
    position = frame.f_lasti
    try:
        name = assigned_name(code, position)
    except Exception: # Bytecode not as expected (e.g., new Python version)
        name = infer_from_source(frame, caller)
    sites[position] = name
    return name

# %% Source line inference (fallback)

cache = {}
max_filecache = 50
def infer_from_source(frame, caller):
    filename = frame.f_code.co_filename
    
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
        
        if filename.startswith('<'):
            # Pseudo-files (e.g., doctest examples) may provide their source
            # through linecache.getlines (which doctest patches).
            try:
                lines = linecache.getlines(filename, frame.f_globals)
            except:
                lines = None
            else:
                if not lines: # No source available (e.g., python -c or exec)
                    lines = None
                else:
                    # Do not cache; pseudo-file names may be reused with different source.
                    lines = list(lines)
                    variable_cache = {}
                    crossout = False
                    filename = None
        else:
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
        caller_source = caller.__func__ if hasattr(caller, '__func__') else caller
        key = (n, caller_source)
        if key in variable_cache:
            return variable_cache[key]
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
            if obj == caller: 
                variable_cache[n] = variable_name
            else:
                variable_cache[n, caller_source] = variable_name = default # No assignment
        else:
            variable_cache[n] = variable_name = default # No assignment
        return variable_name
    except:
        if crossout: lines[n] = None
        variable_cache[n] = default
        return default # Could not infer name