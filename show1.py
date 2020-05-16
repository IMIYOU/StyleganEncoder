import os
import pickle
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator
import matplotlib.pyplot as plt
import glob
 
# 预训练好的网络模型，来自NVIDIA
Model = 'stylegan2-ffhq-config-f.pkl'
_Gs_cache = dict()
 
# 加载StyleGAN已训练好的网络模型
def load_Gs(model):
    if model not in _Gs_cache:
        model_file = glob.glob(Model)
        if len(model_file) == 1:
            model_file = open(model_file[0], "rb")
        else:
            raise Exception('Failed to find the model')
 
        _G, _D, Gs = pickle.load(model_file)
        _Gs_cache[model] = Gs
    return _Gs_cache[model]
 
 
# 将真实人脸图片对应的latent与改变人脸特性/表情的向量相混合，调用generator生成人脸的变化图片
def move_and_show(generator,latent_vector, direction_name, coeffs):
    direction = np.load('latent_directions/'+direction_name+'.npy')
    isExists=os.path.exists(os.path.join(config.generated_dir, direction_name))
    if not isExists:
        os.mkdir(os.path.join(config.generated_dir, direction_name))

    # 调用coeffs数组，生成一系列的人脸变化图片
    for i, coeff in enumerate(coeffs):
        new_latent_vector = latent_vector.copy()
        # 人脸latent与改变人脸特性/表情的向量相混合，只运算前8层（一共18层）
        new_latent_vector[:8] = (latent_vector + coeff*direction)[:8]
        #
        new_latent_vector = new_latent_vector.reshape((1, 18, 512))
        # 将向量赋值给generator
        generator.set_dlatents(new_latent_vector)
        # 调用generator生成图片
        new_person_image = generator.generate_images()[0]
        # 画图，1024x1024
        canvas = PIL.Image.new('RGB', (1024, 1024), 'white')
        canvas.paste(PIL.Image.fromarray(new_person_image, 'RGB'), ((0, 0)))
    
        # 将生成的图像保存到文件
        canvas.save(os.path.join(config.generated_dir, direction_name,direction_name)+str(i)+".png")
        
def main():
    # 初始化
    tflib.init_tf()
    # 调用预训练模型
    Gs_network = load_Gs(Model)
    generator = Generator(Gs_network, batch_size=1, randomize_noise=False)
 
    # 读取对应真实人脸的latent，用于图像变化，qing_01.npy可以替换为你自己的文件名
    os.makedirs(config.dlatents_dir, exist_ok=True)
    target = np.load(os.path.join(config.dlatents_dir, '1_01_1.npy'))
 

    # 混合人脸和变化向量，生成变化后的图片
    #move_and_show(generator, target, "age", [-6, -4, -3, -2, 0, 2, 3, 4, 6])
    #move_and_show(generator, target, "angle_horizontal", [-6, -4, -3, -2, 0, 2, 3, 4, 6])
    #move_and_show(generator, target, gender, [-6, -4, -3, -2, 0, 2, 3, 4, 6])
    #move_and_show(generator, target, "eyes_open", [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3])
    #move_and_show(generator,target, glasses, [-6, -4, -3, -2, 0, 2, 3, 4, 6])
    #move_and_show(generator,target, smile, [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3])
 
if __name__ == "__main__":
    main()