from neuralintents import GenericAssistance
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import mplfinance as mpf


import pickle, sys, datetime as dt

def myfunction():
    pass

mappings = {
    '': myfunction()
}


Zed = GenericAssistance('intent.json', intent_method=mappings)

Zed.tran_model()

Zed.request( )
