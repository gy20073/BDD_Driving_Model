import os
import sys
import subprocess
from ffprobe import FFProbe


def probe_file(filename):
    '''
        Probes the media file for metadata on length of the video (in seconds)
    '''
    cmnd = ['ffprobe', '-show_format', '-pretty', filename]
    p = subprocess.Popen(cmnd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #print filename
    out, err =  p.communicate()
    duration = out.split('\n')
    whole_time = 0
    if len(duration)>8:
        time = duration[7].split(':')
        hour =  time[-3].split('=')[1]
        minute = time[-2]
        second = time[-1]
        # Saving the time length (in seconds) of the video into the variable 'whole_time'
        whole_time = float(hour)*3600 + float(minute)*60 + float(second)
    return whole_time
    


if __name__ == '__main__':
    '''
        Terminal command: python [dataset path]
    '''
    dataset_path = sys.argv[1]
    output_index = os.path.join(dataset_path, "video_filtered_38_60.txt")

    info_path = os.path.join(dataset_path, "info")
    video_path = os.path.join(dataset_path, "videos")

    count = 0

    for item in os.listdir(video_path):
        if item.endswith(".mov"):
            # now we get a video file
            this_video = os.path.join(video_path, item)
            this_info = os.path.join(info_path, item.split(".")[0]+".json")
            if not os.path.exists(this_info):
                print(this_video, this_info, "the corresponding info does not exists")
                continue
            # otherwise there is a pair
            count = count + 1
            if count % 1000 == 0:
                print(count)
            duration = probe_file(this_video.strip())
            # print duration
            if 60 > duration > 38:
                with open(output_index, "a") as myfile:
                    myfile.write(os.path.abspath(this_video)+"\n")
