import json
from pprint import pprint
import numpy as np
import math

def read_json(json_path, video_filename):
    with open(json_path) as data_file:    
        seg = json.load(data_file)

    locs = seg['locations']
    loc2nparray = lambda locs, key: np.array([x[key] for x in locs]).ravel()
    res = {}
    bad_video_c = 0
    bad_video_t = 0
    bad_video_same = 0
    for ifile, f in enumerate(locs):
        if int(f['course']) == -1 or int(f['speed']) == -1:
            bad_video_c += 1   # Changed for interpolation
            if bad_video_c >= 3:
                break
        if ifile != 0:
            if int(f['timestamp'])-prev_t > 1100:
                bad_video_t = 1
                break
            if abs(int(f['timestamp']) - int(prev_t)) < 1:
                bad_video_same = 1
                break
        prev_t = f['timestamp']
    if bad_video_c >= 3:
        print('This is a bad video because course or speed is -1', json_path, video_filename)
        return None
    if bad_video_t:
        print('This is a bad video because time sample not uniform', json_path, video_filename)
        return None
    if len(locs)==0:
        print('This is a bad video because no location data available', json_path, video_filename)
        return None
    if bad_video_same:
        print('This is a bad video because same timestamps', json_path, video_filename)
        return None
    
    for key in locs[0].keys():
        res[key]=loc2nparray(locs, key)

    # add the starting time point and ending time point as well
    res['startTime'] = seg['startTime']
    res['endTime'] = seg['endTime']

    if res['timestamp'][0] - res['startTime'] > 2000:
        print('This is bad video because starting time too far ahead', json_path, video_filename)
        return None

    if res['endTime'] - res['timestamp'][-1] > 2000:
        print('This is bad video because ending time too far ahead', json_path, video_filename)
        return None

    return res


def fill_missing_speeds_and_courses(values, show_warning):
    l = len(values)
    for i in range(l):
        if values[i] == -1:
            if show_warning:
                print("Warning: course==-1 appears, previous computation might not be reliable")
            if i == (l-1):
                values[i] = values[i-1]
            else:
                if values[i+1] == -1:
                    return None
                values[i] = values[i+1]
    return values


def get_interpolated_speed_xy(res, hz=15):     
    def vec(speed, course):
        t = math.radians(course)
        return np.array([math.sin(t)*speed, math.cos(t)*speed])
    
    course = res['course']
    speed0 = res['speed']
    # first convert to speed vecs
    l=len(course)
    speed = np.zeros((l, 2), dtype = np.float32)
    for i in range(l):
        # interpolate when the number of missing speed is small
        speed0 = fill_missing_speeds_and_courses(speed0, False)
        course = fill_missing_speeds_and_courses(course, True)
        if (speed0 is None) or (course is None):
            return None

        speed[i,:] = vec(speed0[i], course[i])

    tot_ms = res['endTime'] - res['startTime']
    # total number of output
    nout = tot_ms * hz // 1000
    out = np.zeros((nout, 2), dtype=np.float32)
    
    # if time is t second, there should be t+1 points
    last_start = 0
    ts = res['timestamp']
    for i in range(nout):
        # convert to ms timestamp
        timenow = i * 1000.0 / hz + res['startTime']  
        
        while (last_start+1 < len(ts)) and (ts[last_start+1] < timenow):
           last_start += 1
 
        if last_start+1 == len(ts):                    
            out[i, :] = speed[last_start, :]           
        elif timenow <= ts[0]:
            out[i, :] = speed[0, :]
        else:
            time1 = timenow - ts[last_start]
            time2 = ts[last_start+1] - timenow
            r1 = time2 / (time1 + time2)
            r2 = time1 / (time1 + time2)
            inter = r1*speed[last_start, :] + r2*speed[last_start+1, :]
            out[i, :] = inter
    return out

def get_interpolated_speed(json_path, video_filename, hz):
    res = read_json(json_path, video_filename)
    if res is None:
        return None
    out = get_interpolated_speed_xy(res, hz)
    return out
