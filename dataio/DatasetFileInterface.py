# -*- coding: utf-8 -*-
# Filename: DatasetInterface.py
from abc import ABCMeta, abstractmethod

#Dataset file interface
class DatasetFileInterface(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def write(self):
        pass
