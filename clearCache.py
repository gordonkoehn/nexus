#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 09:31:10 2022

@author: gordonkoehn

Cleares the cache files generated during code generation of brian2. 
See Brian2 documentation pg. : 461

"""

from brian2.__init__ import clear_cache

brian2.__init__.clear_cache('cython')