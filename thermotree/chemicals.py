# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 09:41:02 2019

@author: yoelr
"""
from .chemical import Chemical

__all__ = ('Chemicals',)

class Chemicals:
    """Create Chemicals object that contains Chemical objects as attributes.

    Parameters
    ----------
    *IDs : str
           Strings should be one of the following [-]:
              * Name, in IUPAC form or common form or a synonym registered in PubChem
              * InChI name, prefixed by 'InChI=1S/' or 'InChI=1/'
              * InChI key, prefixed by 'InChIKey='
              * PubChem CID, prefixed by 'PubChem='
              * SMILES (prefix with 'SMILES=' to ensure smiles parsing)
              * CAS number
        
    """
    @classmethod
    def combine(cls, chemicals):
        """Return a Chemicals object from an iterable of Chemical objects."""
        self = cls.__new__(cls)
        setfield = setattr
        for c in chemicals: setfield(self, c.ID, c)
        return self
    
    def __init__(self, *IDs):
        setfield = setattr
        for n in IDs: setfield(self, n, Chemical(n))
    
    @property
    def IDs(self):
        return tuple(self.__dict__)
    
    def __len__(self):
        return len(self.__dict__)
    
    def __contains__(self, ID):
        return ID in self.__dict__
    
    def __iter__(self):
        yield from self.__dict__.values()
      
    def extend(self, chemicals):
        """Extend with more Chemicals."""
        setfield = setattr
        for c in chemicals: 
            setfield(self, c.ID, c)
    
    def __repr__(self):
        IDs = [repr(i) for i in self.IDs]
        return f'<{type(self).__name__}({", ".join(IDs)})>'