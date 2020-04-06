
import cv2
import os
import numpy as np
from PIL import Image
import paddlehub as hub



def CutVideo2Image(video_path, img_path):
    cap = cv2.VideoCapture(video_path)
    index = 0
    while(True):
        ret,frame = cap.read() 
        if ret:
            cv2.imwrite('video/frame/%d.jpg'%index, frame)
            # img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            # imgs.append(img_rgb)
            index += 1
        else:
            break
    cap.release()
    print('Video cut finish, all %d frame' % index)


def GetHumanSeg(in_path, out_path):
    # load model
    module = hub.Module(name="deeplabv3p_xception65_humanseg")
    # config
    frame_path = in_path
    test_img_path = [os.path.join(frame_path, fname) for fname in os.listdir(frame_path)]
    input_dict = {"image": test_img_path}

    results = module.segmentation(data=input_dict, output_dir=out_path)


def BlendImg(fore_image, base_image, output_path):
    """
    将抠出的人物图像换背景
    fore_image: 前景图片，抠出的人物图片
    base_image: 背景图片
    """
    # 读入图片
    base_image = Image.open(base_image).convert('RGB')
    fore_image = Image.open(fore_image).resize(base_image.size)

    # 图片加权合成
    scope_map = np.array(fore_image)[:,:,-1] / 255
    scope_map = scope_map[:,:,np.newaxis]
    scope_map = np.repeat(scope_map, repeats=3, axis=2)
    res_image = np.multiply(scope_map, np.array(fore_image)[:,:,:3]) + np.multiply((1-scope_map), np.array(base_image))
    
    #保存图片
    res_image = Image.fromarray(np.uint8(res_image))
    res_image.save(output_path)

def BlendHumanImg(in_path, screen_path, out_path):
    humanseg_png = [filename for filename in os.listdir(in_path)]
    for i, img in enumerate(humanseg_png):
        img_path = os.path.join(in_path + '%d.png' % (i))
        output_path_img = out_path + '%d.png' % i
        BlendImg(img_path, screen_path, output_path_img)
   

def init_canvas(width, height, color=(255, 255, 255)):
    canvas = np.ones((height, width, 3), dtype="uint8")
    canvas[:] = color
    return canvas

def GetGreenScreen(width, height, out_path):
    canvas = init_canvas(width, height, color=(0, 255, 0))
    cv2.imwrite(out_path, canvas)


def CombVideo(in_path, out_path, size):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path,fourcc, 30.0, size)
    files = os.listdir(in_path)

    for i in range(len(files)):
        img = cv2.imread(in_path + '%d.png' % i)
        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        # img = cv2.resize(img, (1280,720))
        out.write(img)#保存帧
    out.release()


# Config
Video_Path = 'video/0.mp4'
FrameCut_Path = 'video/frame/'
FrameSeg_Path = 'video/frame_seg/'
FrameCom_Path = 'video/frame_com/'
GreenScreen_Path = 'video/green.jpg'
ComOut_Path = 'output.mp4'

if __name__ == "__main__":

    # 第一步：视频->图像
    if not os.path.exists(FrameCut_Path):
        os.mkdir(FrameCut_Path)     
        CutVideo2Image(Video_Path, FrameCut_Path)

    # 第二步：抠图
    if not os.path.exists(FrameSeg_Path):
        os.mkdir(FrameSeg_Path) 
        GetHumanSeg(FrameCut_Path, FrameSeg_Path)

    # 第三步：生成绿幕并合成
    if not os.path.exists(GreenScreen_Path):
        GetGreenScreen(1920, 1080, GreenScreen_Path)
    
    if not os.path.exists(FrameCom_Path):
        os.mkdir(FrameCom_Path) 
        BlendHumanImg(FrameSeg_Path, GreenScreen_Path, FrameCom_Path)

    # 第四步：合成视频
    if not os.path.exists(ComOut_Path):
        CombVideo(FrameCom_Path, ComOut_Path, (1920, 1080))
