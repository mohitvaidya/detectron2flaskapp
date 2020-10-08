#  lib imports
import re
from tqdm import tqdm
from timer import timer
from os.path import isfile, join, isdir
import ffmpeg


# Main utility for trim and duration reduction

def pre_process(video_id = None, trim_duration=10,fps=1 ):
    '''
    Parameters
    ----------
    video_id : e.g. 11234, should not include any extensions
    basepath : absolute or relative path, must not end with slash("/"), e.g. /users/mnt
                can be same as current work dir
    video_folder : Meta videos should be stored inside this folder, must followed by basepath, 
                    e.g. /user/mnt/{video_folder}
    out_folder = videos output should be stored inside this folder, must followed by basepath, 
                    e.g. /user/mnt/{out_folder}

    '''

    if isfile(f'./{video_id}_.mp4'):
        print(f'file already exists')
        pass
    else: 
        # os.remove(f'{basepath}/{video_folder}/{video_id}.mp4')
        (
            ffmpeg
            .input(f'/app/{video_id}.mp4')
            .filter('trim', duration= trim_duration)    
            .filter('fps', fps=fps, round='up')
            .output(f'/app/{video_id}_.mp4')
            .run(overwrite_output= True)
        )
        print("process finished")

# if __name__=='__main__':
#     preProcess(video_id='1335', basepath='/data1/code_base/mnt_data/ODbatch/', video_folder='vids', out_folder='outf')
