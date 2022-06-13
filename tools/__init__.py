import sys
import pickle
import datetime

def print_progress(message, rate):
    '''Pring progress'''
    if rate < 0: rate = 0
    if rate > 1: rate = 1
    percent = rate*100
    sys.stdout.write('\r')
    sys.stdout.write('{} {:.2f} % [{:<50s}]'.format(message, percent, '=' * int(percent / 2)))
    sys.stdout.flush()

def save(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

def load(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


TOOLS_CURRENT_TIME = None

def tic():
    global TOOLS_CURRENT_TIME
    TOOLS_CURRENT_TIME = datetime.datetime.now()

def toc(return_timespan=False):
    global TOOLS_CURRENT_TIME
    if return_timespan:
        return datetime.datetime.now() - TOOLS_CURRENT_TIME
    else:
        print('Elapsed:', datetime.datetime.now() - TOOLS_CURRENT_TIME)