import os
import glob
import socket

if socket.gethostname() == 'user-ubuntu':
    mode = 'lab'
elif socket.gethostname() == 'ubuntu':
    mode = 'lab'
elif socket.gethostname() == 'sd-bjpg-hg27.yz02':
    mode = 'kwai27'
elif socket.gethostname() == 'bjfk-hg29.yz02':
    mode = 'kwai29'
else:
    print('new server!')

if mode == 'kwai27':
    prefix1 = '/home/web_server/zhouhuanxiang/disk/vimeo/vimeo_septuplet/sequences'
    prefix2 = '/home/web_server/zhouhuanxiang/disk/vimeo/vimeo_septuplet/sequences_blocky32'
    ffmpeg = '/usr/local/share/ffmpeg_qlh/bin/ffmpeg '
elif mode == 'kwai29':
    prefix = '/home/web_server/zhouhuanxiang/disk'
    ffmpeg = 'ffmpeg '
else:
    prefix1 = '/home1/zhx/disk/vimeo/vimeo_septuplet/sequences'
    prefix2 = '/home1/zhx/disk/vimeo/vimeo_septuplet/sequences_blocky'
    ffmpeg = 'ffmpeg '

clips = glob.glob(os.path.join(prefix1, '*', '*'))
clips_blocky = [i.replace(prefix1, prefix2) for i in clips]

for i, (clip, clip_blocky) in enumerate(zip(clips, clips_blocky)):
    # if i == 10:
    #     break
    os.system('{} -i {}/im%d.png -c:v libx265 -x265-params qp=32:bframes=0:no-sao:loopfilter=0 /home/web_server/zhouhuanxiang/test.mp4'.format(ffmpeg, clip))
    os.makedirs(clip_blocky, exist_ok=True)
    os.system('{} -i /home/web_server/zhouhuanxiang/test.mp4 {}/im%d.png'.format(ffmpeg, clip_blocky))
    os.system('rm /home/web_server/zhouhuanxiang/test.mp4')
    print(i)


'''

ffmpeg -i im%d.png -c:v libx265 -x265-params qp=37:bframes=0:no-sao:loopfilter=0 /home1/zhx/test.mp4

ffmpeg -i /home1/zhx/test.mp4 test/im%d.png

'''

