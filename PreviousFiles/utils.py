import time
import json
import nni


# report the runtime of a function
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f'{func.__name__.upper()} took {end - start} seconds')
        return result

    return wrapper


# report to a log file
def log(func):
    def wrapper(*args, **kwargs):
        with open('log.txt', 'a') as f:
            f.write(f'{func.__name__.upper()} was called at {time.ctime()}\n')
        return func(*args, **kwargs)

    return wrapper


# read a json file
def read_json(file_name):
    with open(file_name, 'rb') as f:
        data = json.load(f)
    return data


# run nni experiment
def run_nni(func):
    def wrapper(*args, **kwargs):
        try:
            params = nni.get_next_parameter()
        except Exception as e:
            print(e)
        res = func(params, *args, **kwargs)
        nni.report_final_result(res)
        return res

    return wrapper
