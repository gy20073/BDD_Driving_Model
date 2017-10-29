from subprocess import call
import os, time
import shutil
import io
import base64
from IPython.display import HTML
import numpy as np
from PIL import ImageDraw, Image, ImageFont
from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import matplotlib
import math
import copy
import itertools
import tensorflow as tf
import subprocess
FLAGS = tf.app.flags.FLAGS
import cv2
#from pylab import *
import pylab
from matplotlib.patches import Wedge
from scipy.ndimage.filters import gaussian_filter
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDrawingArea
from matplotlib.patches import FancyArrowPatch

def images2video_highqual(frame_rate,
                 name="temp_name", dir_name="temp_dir"):

    # make dir if not exists
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    pwd = os.getcwd()
    os.chdir(dir_name)

    print("converting to video")
    video_name = name+'.mp4'
    cmd = "ffmpeg -y -f image2 -r " + str(frame_rate) + " -pattern_type glob -i '*.png' -crf 5 -preset veryslow " + \
          "-threads 16 -vcodec libx264  -pix_fmt yuv420p " + video_name
    call(cmd, shell=True)

    call("rm *.png", shell=True)
    os.chdir(pwd)

    return os.path.join(dir_name, video_name)

def images2video(images, frame_rate,
                 name="temp_name", dir_name="temp_dir", highquality=True):
    images = np.uint8(images)
    shape = images.shape
    assert (len(shape) == 4)
    assert (shape[3] == 3 or shape[3] == 1)

    # make dir if not exists
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    pwd = os.getcwd()
    os.chdir(dir_name)

    # write out images
    print("writing images")
    for i in range(shape[0]):
        j = Image.fromarray(images[i, :, :, :])
        j.save("%05d.jpeg" % i, "jpeg", quality=93)

    print("converting to video")
    video_name = name+'.mp4'

    quality_str = '16' if highquality else '28'
    cmd = "ffmpeg -y -f image2 -r " + str(frame_rate) + " -pattern_type glob -i '*.jpeg' -crf "+quality_str+" -preset veryfast " + \
          "-threads 16 -vcodec libx264  -pix_fmt yuv420p " + video_name
    call(cmd, shell=True)

    call("rm *.jpeg", shell=True)
    os.chdir(pwd)

    return os.path.join(dir_name, video_name)

def play_video(path):
    video = io.open(path, 'r+b').read()
    encoded = base64.b64encode(video)
    return HTML(data='''<video alt="test" controls>
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                 </video>'''.format(encoded.decode('ascii')))

def visualize_images(images, frame_rate,
                     name="temp_name", dir_name="temp_dir",delete_temp=True):
    path = images2video(images, frame_rate, name, dir_name)
    out = play_video(path)
    if delete_temp:
        assert not("*" in dir_name)
        shutil.rmtree(dir_name)
    return out

def write_text_on_image(image, string,
                        lines=[],
                        fontsize=30,
                        lines_color=[]):
    shape = image.shape
    assert (len(shape) == 3)
    assert (shape[-1] == 3 or shape[-1] == 1)

    image = np.uint8(image)
    j = Image.fromarray(image)
    draw = ImageDraw.Draw(j)
    # font = ImageFont.load_default().font
    #font = ImageFont.truetype("/usr/share/fonts/truetype/inconsolata/Inconsolata.otf", fontsize)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", fontsize)

    if isinstance(string, list):
        for s in string:
            draw.text(s[0], s[1], s[2], font=font)
    else:
        draw.text((0, 0), string, (255, 0, 0), font=font)

    for line in lines:
        draw.line(line, fill=128, width=1)
    for line in lines_color:
        draw.line(line[0], fill=line[1], width=1)

    return np.array(j)

def egomotion2animation(ego):
    # ego is a egomotion matrix, with nframes * previous frames * 3
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    line = ax.plot([], [], '.', zs=[])
    line = line[0]

    def get_range(ego, axis):
        data = ego[:, :, axis]
        data = np.reshape(data, [-1])
        return [np.min(data), np.max(data)]

    ax.axis(get_range(ego, 0) + get_range(ego, 1))
    zrange = get_range(ego, 2)
    ax.set_zlim(zrange[0], zrange[1])

    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return line,

    # animation function.  This is called sequentially
    def animate(i):
        line.set_data(ego[i, :, 0], ego[i, :, 1])
        line.set_3d_properties(ego[i, :, 2])
        return line,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=ego.shape[0], blit=True)
    plt.close(anim._fig)
    return anim

def animation2HTML(anim, frame_rate):
    print("animaiton to video...")
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, fps=frame_rate,
                      extra_args=['-vcodec', 'libx264',
                                  '-pix_fmt', 'yuv420p',
                                  '-crf', '28',
                                  '-preset', 'veryfast'])

            video = io.open(f.name, 'r+b').read()
            encoded = base64.b64encode(video)
            return HTML(data='''<video alt="test" controls>
                            <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                         </video>'''.format(encoded.decode('ascii')))

def visualize_egomotion(ego, frame_rate):
    anim = egomotion2animation(ego)
    return animation2HTML(anim, frame_rate)

def vis_reader(tout, frame_rate, j=0):
    decoded, isvalid, ego, name, isstop = tout

    images = decoded[j, :, :, :, :]
    images_txt = np.zeros_like(images)
    this_stop = isstop[j]
    this_valid = isvalid[j]

    for i in range(images.shape[0]):
        stop_str = {1: "STOP",
                    0: "GO",
                    -1: "UNKNOWN"}[this_stop[i]]
        valid_str = {0: "Egomotion=Invalid",
                     1: "Egomotion=Valid"}[this_valid[i]]
        showing_str = stop_str + "\n" + valid_str
        # showing_str = stop_str

        images_txt[i, :, :, :] = write_text_on_image(images[i, :, :, :], showing_str)
    print("showing visualization for video %s" % name[0])
    return visualize_images(images_txt, frame_rate)

def move_to_line(move, h, w, multiplier = 10):
    m = copy.deepcopy(move)
    m[1] *= multiplier
    m = [m[1] * math.sin(m[0]), m[1]*math.cos(m[0])]
    return [w / 2, h, w/2+m[0], h-m[1]]

def draw_bar_on_image(image, bar_left_top, fraction, fill=(0,0,0,128), height=20, length=120):
    image = np.uint8(image)
    j = Image.fromarray(image)
    draw = ImageDraw.Draw(j)

    l = bar_left_top
    draw.rectangle([l, (l[0]+int(length*fraction), l[1]+height)], fill=fill)

    return np.array(j)

def vis_reader_stop_go(tout, prediction,frame_rate,  j=0, save_visualize = False, dir_name="temp", provider="nexar_large_speed"):
    #out_of_date, won't do stop go any more
    decoded = tout[0]
    speed = tout[1]
    name = tout[2]
    highres = tout[3]

    isstop = tout[4]
    turn = tout[5]
    locs = tout[6]
    decoded = highres

    turn = turn[j, :, :]
    locs = locs[j, :, :]

    images = decoded[j, :, :, :, :]
    images_txt = np.zeros_like(images)
    
    stop = isstop[j, :]
    speed = speed[j, :, :]

    for i in range(images.shape[0]):
        showing_str = "STOP" if prediction[i] == 1 else "GO!"
        showing_str += "\n" + str(np.linalg.norm(speed[i, :]))
        showing_str += "\n" + "GT: STOP" if stop[i] == 1 else "\nGT: GO!"
        images_txt[i, :, :, :] = write_text_on_image(images[i, :, :, :],
                                                     showing_str)
    print("showing visualization for video %s" % name[0])
    #vis_speed(speed, frame_rate)
    if save_visualize:
        _, short_name = os.path.split(name[j])
        short_name = short_name.split(".")[0]
        return visualize_images(images_txt, frame_rate,
                                name=short_name,
                                dir_name=dir_name,
                                delete_temp=False)
    else:
        return visualize_images(images_txt, frame_rate)

def vis_discrete(tout, predict, frame_rate,
                 j=0, save_visualize=False, dir_name="temp"):
    import data_providers.nexar_large_speed as provider
    int2str = provider.MyDataset.turn_int2str
    
    # city_data and only_seg are mutually exclusive, actually one flag is enough
    if FLAGS.city_data == 1:
        decoded = tout[0]
        speed = tout[1]
        name = tout[2]
        isstop = tout[5]
        turn = tout[6]
        locs = tout[7]
    elif FLAGS.only_seg == 1:
        decoded = tout[0]
        speed = tout[1]
        name = tout[2]
        isstop = tout[6]
        turn = tout[7]
        locs = tout[8]
    else:
        decoded = tout[0]
        speed = tout[1]
        name = tout[2]
        highres = tout[3]

        isstop = tout[4]
        turn = tout[5]
        locs = tout[6]
        decoded = highres

    images = copy.deepcopy(decoded[j, :, :, :, :])
    _, hi, wi, _ = images.shape
    locs = locs[j, :, :]
    turn = turn[j, :, :]

    for i in range(images.shape[0]):
        # the ground truth course and speed
        showing_str = "speed: %.1f m/s \ncourse: %.2f degree/s" % \
                      (locs[i, 1], locs[i, 0]/math.pi*180)
        for k in range(4):
            showing_str += "\n"+int2str[k]
        gtline = move_to_line(locs[i,:], hi, wi)

        FontHeight=18
        FontWidth =8

        for k in range(4):
            images[i, :, :, :] = draw_bar_on_image(images[i,:,:,:],
                                                       (FontWidth*14, FontHeight*(2+k)),
                                                       fraction = turn[i, k],
                                                       fill=(255, 0, 0, 128),
                                                       height=FontHeight * 2 // 3,
                                                       length=FontWidth * 4)
            images[i, :, :, :] = draw_bar_on_image(images[i, :, :, :],
                                                   (FontWidth * 20, FontHeight * (2 + k)),
                                                   fraction=predict[i, k],
                                                   fill=(0, 255, 0, 128),
                                                   height=FontHeight * 2 // 3,
                                                   length=FontWidth * 4)

        images[i, :, :, :] = write_text_on_image(images[i, :, :, :],
                                                 showing_str,
                                                 [gtline],
                                                 fontsize=15)
    print("showing visualization for video %s" % name[j])
    if save_visualize:
        _, short_name = os.path.split(name[j])
        short_name = short_name.split(".")[0]
        for i in range(10):
            this_name = short_name + "_" + str(i)
            if not os.path.isfile(os.path.join(dir_name,this_name+'.mp4')):
                break

        return visualize_images(images, frame_rate,
                                name=this_name,
                                dir_name=dir_name,
                                delete_temp=False)
    else:
        return visualize_images(images, frame_rate)

def vis_discrete_simplified(tout, predict, frame_rate,
                 j=0, save_visualize=False, dir_name="temp"):
    import data_providers.nexar_large_speed as provider
    int2str = provider.MyDataset.turn_int2str

    decoded = tout[0]
    speed = tout[1]
    name = tout[2]
    highres = tout[3]

    isstop = tout[4]
    turn = tout[5]
    locs = tout[6]
    decoded = highres

    images = copy.deepcopy(decoded[j, :, :, :, :])
    _, hi, wi, _ = images.shape
    locs = locs[j, :, :]
    turn = turn[j, :, :]

    for i in range(images.shape[0]):
        # the ground truth course and speed
        showing_str = ""
        for k in range(4):
            showing_str += int2str[k] + "\n"

        FontHeight = 18
        FontWidth = 8

        for k in range(4):
            images[i, :, :, :] = draw_bar_on_image(images[i, :, :, :],
                                                   (FontWidth * 14, FontHeight * k),
                                                   fraction=turn[i, k],
                                                   fill=(255, 0, 0, 128),
                                                   height=FontHeight * 2 // 3,
                                                   length=FontWidth * 4)
            images[i, :, :, :] = draw_bar_on_image(images[i, :, :, :],
                                                   (FontWidth * 20, FontHeight * k),
                                                   fraction=predict[i, k],
                                                   fill=(0, 255, 0, 128),
                                                   height=FontHeight * 2 // 3,
                                                   length=FontWidth * 4)

        images[i, :, :, :] = write_text_on_image(images[i, :, :, :],
                                                 showing_str,
                                                 fontsize=15)
    print("showing visualization for video %s" % name[j])
    if save_visualize:
        _, short_name = os.path.split(name[j])
        short_name = short_name.split(".")[0]
        for i in range(10):
            this_name = short_name + "_" + str(i)
            if not os.path.isfile(os.path.join(dir_name, this_name + '.mp4')):
                break

        return visualize_images(images, frame_rate,
                                name=this_name,
                                dir_name=dir_name,
                                delete_temp=False)
    else:
        return visualize_images(images, frame_rate)

def generate_meshlist(arange1, arange2):
    return np.dstack(np.meshgrid(arange1, arange2, indexing='ij')).reshape((-1,2))

def draw_sector(image,
                predict,
                car_stop_model,
                course_delta = 0.5 / 180 * math.pi,
                speed_delta=0.3,
                pdf_multiplier=255,
                speed_multiplier = 5,
                h=360, w=640,
                max_speed=30,
                uniform_speed=False,
                consistent_vis=(False, 1e-3, 1e2),
                has_alpha_channel=False):

    course_samples = np.arange(-math.pi / 2-course_delta,
                               math.pi / 2+course_delta,
                               course_delta)
    speed_samples = np.arange(0, max_speed+speed_delta, speed_delta)

    total_pdf = car_stop_model.continous_pdf([predict],
                                                generate_meshlist(course_samples, speed_samples),
                                                "multi_querys")
    total_pdf = np.reshape(total_pdf, (len(course_samples), len(speed_samples)))

    if uniform_speed:
        total_pdf = total_pdf / np.sum(total_pdf, axis=1, keepdims=True)

    speed_scaled = max_speed * speed_multiplier
    # potential xy positions to be filled
    xy = generate_meshlist(np.arange(w / 2 - speed_scaled, w / 2 + speed_scaled),
                           np.arange(h - speed_scaled, h))
    # filter out invalid speed
    v=np.stack((xy[:,0]-w/2, h-xy[:,1]), axis=1)
    speed_norm = np.sqrt(v[:,0]**2 + v[:,1]**2) *(1.0/speed_multiplier)

    valid_speed = np.less(speed_norm, max_speed)

    xy = xy[valid_speed, :]
    speed_norm=speed_norm[valid_speed]
    v=v[valid_speed]
    course_norm = np.arctan(1.0*v[:, 0] / v[:, 1])

    # search the course and speed
    icourse = np.searchsorted(course_samples, course_norm)
    ispeed = np.searchsorted(speed_samples, speed_norm)

    green_portion = 1
    total = total_pdf[icourse, ispeed]
    if consistent_vis[0] == False:
        total_max = np.amax(total)
        total = total / total_max * 255*green_portion
    else:
        # consistent visualization between methods
        MIN = consistent_vis[1]
        MAX = consistent_vis[2]
        total = np.maximum(MIN, total)
        total = np.minimum(MAX, total)
        #total = np.log(total) # map to log(MIN) to log(MAX)
        #total = (total -np.log(MIN)) / (np.log(MAX) - np.log(MIN)) * 255
        total = (total - MIN) / (MAX - MIN)
        total = np.sqrt(total)
        total = total * 255

    # assign to image
    image[xy[:, 1], xy[:, 0], :] *= (1-green_portion)
    image[xy[:, 1], xy[:, 0], 1] += total
    if has_alpha_channel:
        image[xy[:, 1], xy[:, 0], 3] = 255

    return image

def vis_continuous(tout, predict, frame_rate, car_stop_model,
                 j=0, save_visualize=False, dir_name="temp", return_first=False, **kwargs):
    decoded = tout[0]
    speed = tout[1]
    name = tout[2]
    highres = tout[3]

    isstop = tout[4]
    turn = tout[5]
    locs = tout[6]
    decoded = highres

    images = copy.deepcopy(decoded[j, :, :, :, :])
    images = images.astype('float64')
    _, hi, wi, _ = images.shape
    locs = locs[j, :, :]

    for i in range(images.shape[0]):
        # the ground truth course and speed
        showing_str = "speed: %.1f m/s \ncourse: %.2f degree/s" % \
                      (locs[i, 1], locs[i, 0] / math.pi * 180)
        gtline = move_to_line(locs[i, :], hi, wi, 10)

        images[i, :, :, :] = draw_sector(images[i, :, :, :],
                    predict[i:(i+1), :],
                    car_stop_model,
                    course_delta=0.3 / 180 * math.pi,
                    speed_delta=0.3,
                    pdf_multiplier=255*10,
                    speed_multiplier=wi/30/3,
                    h=hi, w=wi,
                    consistent_vis=(True, 1e-5, 0.3))

        # get the MAP prediction
        map = car_stop_model.continous_MAP([predict[i:(i+1), :]])
        mapline = move_to_line(map.ravel(), hi, wi, 10)

        # swap the shorter line to the latter, avoid overwriting
        lines_v = [(gtline, (255,0,0)), (mapline, (0, 0, 255))]
        if locs[i, 1] < map.ravel()[1]:
            lines_v = [lines_v[1], lines_v[0]]

        images[i, :, :, :] = write_text_on_image(images[i, :, :, :],
                                                 showing_str,
                                                 lines_color=lines_v,
                                                 fontsize=15)

    print("showing visualization for video %s" % name[j])

    if return_first:
        return images[0, :, :, :].astype(np.uint8)

    if save_visualize:
        _, short_name = os.path.split(name[j])
        short_name = short_name.split(".")[0]
        return visualize_images(images, frame_rate,
                                name=short_name,
                                dir_name=dir_name,
                                delete_temp=False)
    else:
        return visualize_images(images, frame_rate)

def vis_continuous_simplified(tout, predict, frame_rate, car_stop_model,
                 j=0, save_visualize=False, dir_name="temp", vis_radius=10):
    decoded = tout[0]
    speed = tout[1]
    name = tout[2]
    highres = tout[3]

    isstop = tout[4]
    turn = tout[5]
    locs = tout[6]
    decoded = highres

    images = copy.deepcopy(decoded[j, :, :, :, :])
    images = images.astype('float64')
    _, hi, wi, _ = images.shape
    locs = locs[j, :, :]
    locs = copy.deepcopy(locs)

    for i in range(images.shape[0]):
        # the ground truth course and speed
        locs[i, 1] = 10.0
        # get the MAP prediction
        map = car_stop_model.continous_MAP([predict[i:(i+1), :]])
        map = map.ravel()
        map[1] = 10.0
        mapline = move_to_line(map, hi, wi, 10)

        # get map2
        map2 = car_stop_model.continous_MAP([predict[i:(i + 1), :]], return_second_best=True)
        map2 = map2.ravel()
        map2[1] = 10.0
        mapline2 = move_to_line(map2, hi, wi, 10)

        showing_str = [
            [(0, 0), "driver's  angular speed: %.2f degree/s" % (locs[i, 0] / math.pi * 180), (255, 0, 0)],
            [(0, 20), "predicted angular speed: %.2f degree/s" % (map[0] / math.pi * 180), (0, 0, 255)]]
        # disable the small str on top first
        showing_str = ""
        showing_str = "speed: %.1f m/s \ncourse: %.2f degree/s" % \
                      (locs[i, 1], locs[i, 0] / math.pi * 180)

        gtline = move_to_line(locs[i, :], hi, wi, 10)

        if FLAGS.is_MKZ_dataset:
            # might be problematic since we enable the normalization
            higher_bound = 0.3
        else:
            higher_bound = 3.0

        images[i, :, :, :] = draw_sector(images[i, :, :, :],
                    predict[i:(i+1), :],
                    car_stop_model,
                    course_delta=0.1 / 180 * math.pi,
                    speed_delta=0.1,
                    pdf_multiplier=255*10,
                    speed_multiplier=int(wi/30/3),
                    h=hi, w=wi,
                    uniform_speed=True,
                    consistent_vis=(True, 1e-5, higher_bound))

        # disable the MAP line first, since many times not the MAP line is considered
        '''
        # swap the shorter line to the latter, avoid overwriting
        lines_v = [(gtline, (255,0,0)), (mapline, (0, 0, 255))]
        if locs[i, 1] < map.ravel()[1]:
            lines_v = [lines_v[1], lines_v[0]]
        '''
        lines_v = [(gtline, (255,0,0)), (mapline, (0,0,255)), (mapline2, (0, 255, 0))]

        images[i, :, :, :] = write_text_on_image(images[i, :, :, :],
                                                 showing_str,
                                                 lines_color=lines_v,
                                                 fontsize=24)
    print("showing visualization for video %s" % name[j])
    if save_visualize:
        _, short_name = os.path.split(name[j])
        short_name = short_name.split(".")[0]
        return visualize_images(images, frame_rate,
                                name=short_name,
                                dir_name=dir_name,
                                delete_temp=False)
    else:
        return visualize_images(images, frame_rate)

# some visualization functions for the speed
def visLoc(locs, label="NotSet"):
    axis = lambda i: [loc[i] for loc in locs]
    import matplotlib.ticker as ticker
    
    fig, ax = plt.subplots()
    #plt.grid(True)
    ax.plot(axis(0), axis(1), 'g^', ms=2)
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    ax.set_xlim(min(xlim[0],ylim[0]) ,max(xlim[1],ylim[1]))
    ax.set_ylim(min(xlim[0],ylim[0]) ,max(xlim[1],ylim[1]))
   
    plt.title("Moving paths from " + label)
    plt.xlabel("West -- East")
    plt.ylabel("South -- North")
    plt.show()

def integral(speed, time0):
    out = np.zeros_like(speed)
    l = speed.shape[0]
    for i in range(l):
        s = speed[i, :]
        if i > 0:
            out[i, :] = out[i - 1, :] + s * time0
    return out

def vis_speed(speed, hz):
    visLoc(integral(speed, 1.0 / hz), "speed and course")


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    # x has shape: #instances * #classes
    maxes = np.max(x, axis=1)
    e_x = np.exp(x - maxes[:, None])

    sums = np.sum(e_x, axis=1)
    return e_x / sums[:, None]

def read_video_file(video_path, HEIGHT, WIDTH):
    # take a video's path and return its decoded contents
    cmnd = ['ffmpeg',
            '-i', video_path,
            '-f', 'image2pipe',
            '-loglevel', 'panic',
            '-pix_fmt', 'rgb24',
            '-vcodec', 'rawvideo', '-']
    pipe = subprocess.Popen(cmnd, stdout=subprocess.PIPE, bufsize=10 ** 7)
    pout, perr = pipe.communicate()
    image_buff = np.fromstring(pout, dtype='uint8')
    if image_buff.size % (HEIGHT*WIDTH):
        print("Height and Width are potentially not correct")
        return None
    image_buff = image_buff.reshape((-1, HEIGHT, WIDTH, 3))

    return image_buff


def vis_discrete_colormap_antialias(tout, predict, frame_rate, j=0, save_visualize=False, dir_name="temp", string_type='image'):
    if FLAGS.only_seg:
        decoded = tout[0]
        speed = tout[1]
        name = tout[2]
        isstop = tout[6]
        turn = tout[7]
        locs = tout[8]
    else:
        decoded = tout[0]
        speed = tout[1]
        name = tout[2]
        highres = tout[3]

        isstop = tout[4]
        turn = tout[5]
        locs = tout[6]
        decoded = highres

    images = copy.deepcopy(decoded[j, :, :, :, :])
    _, hi, wi, _ = images.shape
    turn = turn[j, :, :]

    def get_color(prob):
        cm = pylab.get_cmap('viridis')  # inferno
        color = cm(prob)  # color will now be an RGBA tuple
        r = color[0] * 255
        g = color[1] * 255
        b = color[2] * 255
        return r, g, b

    def clamp(x):
        x = float(x)
        return max(0, min(x, 1))

    def add_to_ada(ada, pos_x, pos_y, radius, angle_s, angle_e, ring_width, color_code, edge_color, alpha_value):
        ada.drawing_area.add_artist(
            Wedge((pos_x, pos_y), radius, angle_s, angle_e, width=ring_width  # , color=color_code#'#DAF7A6'
                  , alpha=alpha_value, antialiased=True, ec=edge_color, fc=color_code))

    def draw_cake(ada, pos_x, pos_y, radius, angle_s, angle_diff, ring_width, color_code, edge_color, alpha_value,
                  share, shift=45):
        angle_s = angle_s + shift
        for i in range(share):
            if (angle_s + (i + 1) * (angle_diff) / share) == 360:
                angle_end = 360
            else:
                angle_end = angle_s + (i + 1) * (angle_diff) / share
            #print(i,'_______________________________________')
            add_to_ada(ada, pos_x, pos_y, radius,
                       angle_s + i * (angle_diff) / share, angle_end,
                       ring_width, color_code=color_code, edge_color=edge_color, alpha_value=alpha_value[i])

    def draw_pile_cake(ada, pos_x, pos_y, radius, angle_s, angle_diff, ring_width, color_code, edge_color, alpha_value,
                       share, x_frac, y_frac, split, fontsize=24, shift=45):
        # draw the black one
        draw_cake(ada, pos_x=pos_x, pos_y=pos_y, radius=radius, angle_s=angle_s, angle_diff=360, ring_width=None,
                  color_code='k', edge_color=None, alpha_value=[0.6], share=1)
        # draw the green one

        draw_cake(ada, pos_x=pos_x, pos_y=pos_y, radius=radius, angle_s=angle_s, angle_diff=360, ring_width=ring_width,
                  color_code=color_code, edge_color='#FFFFFF', alpha_value=alpha_value, share=4)
        # draw the white edge
        draw_cake(ada, pos_x=pos_x, pos_y=pos_y, radius=radius, angle_s=angle_s, angle_diff=360, ring_width=ring_width,
                  color_code='none', edge_color='#FFFFFF', alpha_value=[1, 1, 1, 1], share=4)
        ada.da.add_artist(
            ax.annotate(split, xy=(x_frac, y_frac), xycoords="axes fraction", fontsize=fontsize, color='w'))

    def draw_cake_type(ada, string_type, action_mean, predict_mean):
        if string_type == 'video':
            draw_pile_cake(ada, pos_x=210, pos_y=70, radius=60, angle_s=0, angle_diff=360, ring_width=30,
                           color_code='#00FF00', edge_color=None, alpha_value=predict_mean, share=1,
                           x_frac=0.513, y_frac=0.895, split='P')
            draw_pile_cake(ada, pos_x=80, pos_y=70, radius=60, angle_s=0, angle_diff=360, ring_width=30,
                           color_code='#00FF00', edge_color=None, alpha_value=action_mean, share=1,
                           x_frac=0.185, y_frac=0.895, split='G')
        elif string_type == 'image':

            draw_pile_cake(ada, pos_x=240, pos_y=70, radius=70, angle_s=0, angle_diff=360, ring_width=40,
                           color_code='#00FF00', edge_color=None, alpha_value=predict_mean, share=1,
                           x_frac=0.580, y_frac=0.89, split='P', fontsize=32)
            draw_pile_cake(ada, pos_x=80, pos_y=70, radius=70, angle_s=0, angle_diff=360, ring_width=40,
                           color_code='#00FF00', edge_color=None, alpha_value=action_mean, share=1,
                           x_frac=0.18, y_frac=0.89, split='G', fontsize=32)

    _, short_name = os.path.split(name[j])
    short_name = short_name.split(".")[0]

    for i in range(images.shape[0]):
        action_mean = [clamp(turn[i, 0]+0.05), clamp(turn[i, 2]+0.05),
                       clamp(turn[i, 1]+0.1), clamp(turn[i, 3]+0.05)]
        predict_mean = [clamp(predict[i, 0]+0.05), clamp(predict[i, 2]+0.05),
                        clamp(predict[i, 1]+0.05), clamp(predict[i, 3]+0.05)]
        fig = plt.figure(figsize=(16, 12))
        ax_original = plt.gca()
        ax_original.set_axis_off()
        ax_original.get_xaxis().set_visible(False)
        ax_original.get_yaxis().set_visible(False)
        plt.imshow(images[i, :, :, :])
        plt.axis('off')
        ax = fig.add_subplot(121, projection='polar')
        ax_2 = fig.add_subplot(122, projection='polar')

        ada = AnchoredDrawingArea(200, 100, 0, 0, loc=2, pad=0., frameon=False)
        draw_cake_type(ada, string_type, action_mean, predict_mean)

        ax.add_artist(ada)
        ax.set_axis_off()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax_2.set_axis_off()
        ax_2.get_xaxis().set_visible(False)
        ax_2.get_yaxis().set_visible(False)

        if not os.path.exists(os.path.join(dir_name,'viz')):
            os.mkdir(os.path.join(dir_name,'viz'))
        if not os.path.exists(os.path.join(dir_name,'viz', short_name+string_type)):
            os.mkdir(os.path.join(dir_name, 'viz', short_name+string_type))
        fig.savefig(os.path.join(dir_name, 'viz', short_name+string_type,'{0:04}.png'.format(i)),
                    bbox_inches='tight', pad_inches = -0.04, Transparent=True, dpi=100)

        print(short_name,' ', i, 'Done!')
        plt.show()
        plt.close()

    images2video_highqual(frame_rate = 3,
                          name=short_name, dir_name=os.path.join(dir_name, 'viz', short_name+string_type))

def vis_continuous_colormap_antialias(tout, predict, frame_rate, car_stop_model,
                 j=0, save_visualize=False, dir_name="temp", vis_radius=10):
    decoded = tout[0]
    speed = tout[1]
    name = tout[2]
    highres = tout[3]

    isstop = tout[4]
    turn = tout[5]
    locs = tout[6]
    decoded = highres

    images = copy.deepcopy(decoded[j, :, :, :, :])
    #images = images.astype('float64')
    _, hi, wi, _ = images.shape
    locs = locs[j, :, :]

    def plot_greens(bin_ends, values, image_width, image_height, radius, driver_action):
        # bins are: [0, bin_ends[0]], [bin_ends[0], bin_ends[1]] ...
        # and the corresponding values to display are: values[0], values[1]
        # the final results are added to ada

        ada = AnchoredDrawingArea(radius * 2, radius, 0, 0, loc=10, pad=0., frameon=False)
        def add_ada_custom(angle_s, angle_e, value, color):
            add_to_ada(ada, radius, -(image_height / 2 - radius / 2), radius, angle_s, angle_e, None, color, value)

        def add_to_ada(ada, pos_x, pos_y, radius, angle_s, angle_e, ring_width, color_code, alpha_value):
            ada.drawing_area.add_artist(
                Wedge((pos_x, pos_y), radius, angle_s, angle_e, width=ring_width, fc=color_code  # '#DAF7A6'
                      ,ec = 'none', alpha=alpha_value, antialiased=True))

        bin_ends = 180 - np.array(bin_ends)
        bin_ends = bin_ends[::-1]
        values = np.array(values)
        values = np.squeeze(values)
        values = values[::-1]

        # add a black background
        add_ada_custom(0, 180, 0.8, "#000000")

        color_shading = "#00FF00"
        for i in range(len(values)):
            #print(bin_ends.shape, '____all____bin_____')
            #print(values.shape, '___all_____values____')
            if i < 5:
                print(bin_ends[i], bin_ends[i + 1], values[i], '________________________')
            add_ada_custom(bin_ends[i], bin_ends[i + 1], values[i], color_shading)

        white_border = 1
        border_color = '#FFFFFF'
        add_to_ada(ada, radius, -(image_height / 2 - radius / 2), radius + white_border, 0, 180, white_border,
                   border_color, 1)

        tick_len = 20
        tick_color = '#FFFFFF'
        tick_width = 1.0 / 2
        for i in range(len(bin_ends)):
            add_to_ada(ada, radius, -(image_height / 2 - radius / 2), radius + white_border,
                       bin_ends[i] - tick_width / 2, bin_ends[i] + tick_width / 2, tick_len, tick_color, 10)

        driver_action = driver_action / 180.0 * math.pi
        start = np.array([radius, -(image_height / 2 - radius / 2) - 2])
        delta = np.array([radius * math.cos(driver_action), radius * math.sin(driver_action)]) * 0.8
        color_driver = "#0000FF"
        ada.drawing_area.add_artist(FancyArrowPatch(start, start + delta, linewidth=2, color=color_driver))

        return ada

    _, short_name = os.path.split(name[j])
    short_name = short_name.split(".")[0]
    for i in range(images.shape[0]):
        # the ground truth course and speed
        locs[i, 1] = 10.0
        # get the MAP prediction
        fig = plt.figure(figsize=(16, 12))
        course_bin, speed_bin = car_stop_model.get_bins()
        course_bin = [-math.pi/2] + course_bin + [math.pi/2]
        course_bin = np.array(course_bin)*180/math.pi + 90
        ax_original = plt.gca()
        ax_original.set_axis_off()
        ax_original.get_xaxis().set_visible(False)
        ax_original.get_yaxis().set_visible(False)
        plt.imshow(images[i, :, :, :])
        plt.axis('off')

        course = softmax(predict[i:(i + 1), 0:FLAGS.discretize_n_bins])
        course = course/np.max(course)

        print(course_bin, course, '!'*10)
        ada2 = plot_greens(course_bin, course, 1280, 501, 200, -locs[i, 0]*180/math.pi+90)
        ax_original.add_artist(ada2)
        plt.show()

        if not os.path.exists(os.path.join(dir_name,'viz')):
            os.mkdir(os.path.join(dir_name,'viz'))
        if not os.path.exists(os.path.join(dir_name,'viz', short_name)):
            os.mkdir(os.path.join(dir_name, 'viz', short_name))
        fig.savefig(os.path.join(dir_name, 'viz', short_name, '{0:04}.png'.format(i)),
                    bbox_inches='tight', pad_inches=-0.04, Transparent=True, dpi=100)
        plt.close()
        print(short_name)

    print("showing visualization for video %s" % name[j])


def vis_continuous_interpolated(tout, predict, frame_rate, car_stop_model,
                 j=0, save_visualize=False, dir_name="temp", vis_radius=10, need_softmax=True, return_first=False):
    decoded = tout[0]
    speed = tout[1]
    name = tout[2]
    highres = tout[3]

    isstop = tout[4]
    turn = tout[5]
    locs = tout[6]
    decoded = highres

    images = copy.deepcopy(decoded[j, :, :, :, :])

    _, hi, wi, _ = images.shape
    locs = locs[j, :, :]

    def gen_mask(bin_ends, values, radius, height, width):
        # convert bin_ends to bin centers
        new_ends = []
        for i in range(len(bin_ends) - 1):
            new_ends.append((bin_ends[i] + bin_ends[i + 1]) / 2)

        # RGBA
        out = np.zeros((height, width, 4), dtype=np.uint8)

        xy = np.dstack(np.meshgrid(np.arange(width / 2 - radius, width / 2 + radius),
                                   np.arange(height - radius, height),
                                   indexing='ij')).reshape((-1, 2))

        # filter out invalid speed
        v = np.stack((xy[:, 0] - width / 2, height - xy[:, 1]), axis=1)
        speed_norm = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2)

        valid_speed = np.less(speed_norm, radius)

        xy = xy[valid_speed, :]
        speed_norm = speed_norm[valid_speed]
        v = v[valid_speed]
        course_norm = np.arccos(1.0 * v[:, 0] / speed_norm)
        course_norm = np.degrees(course_norm)

        value = np.interp(course_norm, new_ends, values)

        out[xy[:, 1], xy[:, 0], 1] = 255 * value
        out[xy[:, 1], xy[:, 0], 3] = 255

        return out

    def plot_greens(bin_ends, values, image_width, image_height, radius, driver_action):
        ada = AnchoredDrawingArea(radius * 2, radius, 0, 0, loc=10, pad=0., borderpad=0., frameon=False)

        def add_to_ada(ada, pos_x, pos_y, radius, angle_s, angle_e, ring_width, color_code, alpha_value):
            ada.drawing_area.add_artist(
                Wedge((pos_x, pos_y), radius, angle_s, angle_e, width=ring_width, fc=color_code  # '#DAF7A6'
                      , ec='none', alpha=alpha_value, antialiased=True))

        bin_ends = 180 - np.array(bin_ends)
        bin_ends = bin_ends[::-1]
        values = np.array(values)
        values = np.squeeze(values)
        values = values[::-1]

        mask = gen_mask(bin_ends, values, radius, image_height, image_width)
        plt.imshow(mask, alpha=0.8)

        white_border = 2
        border_color = '#FFFFFF'
        add_to_ada(ada, radius, -(image_height / 2 - radius / 2), radius + white_border, 0, 180, white_border + 1,
                   border_color, 1)

        tick_len = 20
        tick_color = '#FFFFFF'
        tick_width = 1.0 / 2
        for i in range(len(bin_ends)):
            if abs(bin_ends[i] - 90) > 10:
                add_to_ada(ada, radius, -(image_height / 2 - radius / 2), radius + white_border,
                           bin_ends[i] - tick_width / 2, bin_ends[i] + tick_width / 2, tick_len, tick_color, 10)

        driver_action = driver_action / 180.0 * math.pi
        start = np.array([radius, -(image_height / 2 - radius / 2)])
        delta = np.array([radius * math.cos(driver_action), radius * math.sin(driver_action)]) * 0.8
        color_driver = "#0000FF"
        ada.drawing_area.add_artist(FancyArrowPatch(start, start + delta, linewidth=2, color=color_driver))

        return ada

    _, short_name = os.path.split(name[j])
    short_name = short_name.split(".")[0]
    for i in range(images.shape[0]):
        # the ground truth course and speed
        locs[i, 1] = 10.0
        # get the MAP prediction

        # TODO, might change based on machine
        DPI = 72
        fig = plt.figure(figsize=(1.0*wi / DPI, 1.0*hi / DPI), dpi=DPI)

        course_bin, speed_bin = car_stop_model.get_bins()
        course_bin = [-math.pi/2] + course_bin + [math.pi/2]
        course_bin = np.array(course_bin)*180/math.pi + 90

        #ax_original = plt.gca()
        ax_original = fig.add_axes([0, 0, 1, 1])
        ax_original.set_axis_off()
        ax_original.get_xaxis().set_visible(False)
        ax_original.get_yaxis().set_visible(False)
        plt.imshow(images[i, :, :, :])
        plt.axis('off')

        if need_softmax:
            course = softmax(predict[i:(i + 1), 0:FLAGS.discretize_n_bins])
        else:
            course = predict[i:(i + 1), 0:FLAGS.discretize_n_bins]
        course = course/np.max(course)

        radius = int(hi/2)
        ada2 = plot_greens(course_bin, course, wi, hi, radius, -locs[i, 0]*180/math.pi+90)
        ax_original.add_artist(ada2)
        #plt.show()

        if not os.path.exists(os.path.join(dir_name,'viz')):
            os.makedirs(os.path.join(dir_name,'viz'))
        if not os.path.exists(os.path.join(dir_name,'viz', short_name)):
            os.mkdir(os.path.join(dir_name, 'viz', short_name))
        fig.savefig(os.path.join(dir_name, 'viz', short_name, '{0:04}.png'.format(i)),
                    bbox_inches='tight', pad_inches=0.0, Transparent=True, dpi=DPI)
        plt.close()
        print(short_name)

    print("showing visualization for video %s" % name[j])
    if return_first:
        path = os.path.join(dir_name, 'viz', name[0], '{0:04}.png'.format(0))
        image = misc.imread(path, mode='RGB')
        return image

from scipy import misc
import matplotlib
def continuous_vis_single_image(image, predict, method="vis_continuous"):
    matplotlib.use('Agg')
    assert len(image.shape)==3
    name = ["single_image"]
    fake_locs = np.array([[[0.0, 0.0]]])
    image = np.reshape(image, (1, 1, image.shape[0], image.shape[1], image.shape[2]))
    tout = (image,
            None,
            name,
            image,
            None,
            None,
            fake_locs)

    dir_name = "temp"
    import models.car_stop_model as car_stop_model

    func = globals()[method]

    image = func(tout, predict,
                 frame_rate=None,
                 car_stop_model=car_stop_model,
                 j=0, save_visualize=None,
                 dir_name=dir_name, vis_radius=None,
                 need_softmax=True, return_first=True,
                 save_video=False)

    return image

# improved version with various speed and yaw rate
def vis_continuous_yang(tout, predict, frame_rate, car_stop_model,
                 j=0, save_visualize=False, dir_name="temp", vis_radius=10, need_softmax=True,
                 return_first=False, save_video=True):
    decoded = tout[0]
    speed = tout[1]
    name = tout[2]
    if FLAGS.city_data:
        seg_image = tout[3]
        highres = tout[4]

        isstop = tout[5]
        turn = tout[6]
        locs = tout[7]
    else:
        highres = tout[3]

        isstop = tout[4]
        turn = tout[5]
        locs = tout[6]
    decoded = highres

    images = copy.deepcopy(decoded[j, :, :, :, :])

    _, hi, wi, _ = images.shape
    locs = locs[j, :, :]

    def gen_mask(predict, radius, height, width, consistent_upper_bound=0.3):
        image = np.zeros((height, width, 4), dtype=np.float64)
        image = draw_sector(image,
                    predict,
                    car_stop_model,
                    course_delta=0.1 / 180 * math.pi,
                    speed_delta=0.1,
                    pdf_multiplier=None,
                    speed_multiplier=radius/30,
                    h=height, w=width,
                    max_speed=30,
                    uniform_speed=False,
                    consistent_vis=(True, 1e-3, consistent_upper_bound),
                    has_alpha_channel=True)

        return image.astype("uint8")

    def plot_line(ada, driver_action, driver_speed, image_height, radius, color_driver="#0000FF", linewidth=2.0):
        # convert the driver action to the language of matplotlib
        driver_action = -driver_action * 180 / math.pi + 90
        # then to radian again
        driver_action = driver_action / 180.0 * math.pi

        start = np.array([radius, -(image_height / 2 - radius / 2)])
        delta = np.array([driver_speed * math.cos(driver_action), driver_speed * math.sin(driver_action)]) * 0.8
        ada.drawing_area.add_artist(FancyArrowPatch(start, start + delta, linewidth=linewidth, color=color_driver))

    def plot_greens(predict, image_width, image_height, radius, locs, MAP, amplify_ratio):
        ada = AnchoredDrawingArea(radius * 2, radius, 0, 0, loc=10, pad=0., borderpad=0., frameon=False)

        def add_to_ada(ada, pos_x, pos_y, radius, angle_s, angle_e, ring_width, color_code, alpha_value):
            ada.drawing_area.add_artist(
                Wedge((pos_x, pos_y), radius, angle_s, angle_e, width=ring_width, fc=color_code  # '#DAF7A6'
                      , ec='none', alpha=alpha_value, antialiased=True))

        # TODO, replace this with a call to draw_sector
        mask = gen_mask(predict, radius, image_height, image_width)
        plt.imshow(mask, alpha=0.8)

        white_border = 2
        border_color = '#FFFFFF'
        add_to_ada(ada, radius, -(image_height / 2 - radius / 2), radius + white_border, 0, 180, white_border + 1,
                   border_color, 1)

        plot_line(ada, locs[0], locs[1]*amplify_ratio, image_height, radius, "#FF0000", 1.0)
        plot_line(ada, MAP[0],  MAP[1]*amplify_ratio,  image_height, radius, "#0000FF", 1.0)

        return ada

    _, short_name = os.path.split(name[j])
    short_name = short_name.split(".")[0]
    for i in range(images.shape[0]):
        # TODO, might change based on machine
        DPI = 72
        fig = plt.figure(figsize=(1.0*wi / DPI, 1.0*hi / DPI), dpi=DPI)

        #ax_original = plt.gca()
        ax_original = fig.add_axes([0, 0, 1, 1])
        ax_original.set_axis_off()
        ax_original.get_xaxis().set_visible(False)
        ax_original.get_yaxis().set_visible(False)
        plt.imshow(images[i, :, :, :])
        plt.axis('off')

        assert need_softmax

        radius = int(hi/2) / 30 * 30

        # TODO: add the MAP prediction.
        ada2 = plot_greens(predict[i:(i+1), :], wi, hi, radius,
                           locs[i, :],
                           car_stop_model.continous_MAP([predict[i:(i + 1), :]]).ravel(),
                           radius / 30.0)
        ax_original.add_artist(ada2)
        #plt.show()

        out_dir_name = dir_name
        if not os.path.exists(out_dir_name):
            os.makedirs(out_dir_name)
        # This line take most of the time, because all the rendering happens here
        fig.savefig(os.path.join(out_dir_name, '{0:05}.png'.format(i)),
                    bbox_inches='tight', pad_inches=-0.1, Transparent=True, dpi=DPI)
        plt.close()
        print(short_name)

    if save_video:
        images2video_highqual(frame_rate=frame_rate, name=short_name, dir_name=dir_name)

    print("showing visualization for video %s" % name[j])
    if return_first:
        path = os.path.join(out_dir_name, '{0:05}.png'.format(0))
        image = misc.imread(path, mode='RGB')
        return image
