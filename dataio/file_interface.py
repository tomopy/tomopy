# -*- coding: utf-8 -*-
# Filename: file_interface.py
from abc import ABCMeta, abstractmethod

# Dataset file interface.
class FileInterface(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def write(self):
        pass
