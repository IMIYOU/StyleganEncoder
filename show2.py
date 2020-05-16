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
 
# pre-trained network.
Model = 'stylegan2-ffhq-config-f.pkl'
 
synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)
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

def move_and_show(generator,latent_vector, direction_name, coeffs):
    # Loading already learned latent directions
    direction = np.load('latent_directions/'+direction_name+'.npy')
    isExists=os.path.exists(os.path.join(config.generated_dir, direction_name))
    if not isExists:
        os.mkdir(os.path.join(config.generated_dir, direction_name))

    for i, coeff in enumerate(coeffs):
        new_latent_vector = latent_vector.copy()
        new_latent_vector[:8] = (latent_vector + coeff*direction)[:8]

        new_latent_vector = new_latent_vector.reshape((1, 18, 512))

        generator.set_dlatents(new_latent_vector)

        new_person_image = generator.generate_images()[0]

        canvas = PIL.Image.new('RGB', (1024, 1024), 'white')

        canvas.paste(PIL.Image.fromarray(new_person_image, 'RGB'), ((0, 0)))

        canvas.save(os.path.join(config.generated_dir, direction_name,direction_name)+str(i)+".png")
 
 
def main():
    tflib.init_tf()
    Gs_network = load_Gs(Model)
    generator = Generator(Gs_network, batch_size=1, randomize_noise=False)
 
    os.makedirs(config.dlatents_dir, exist_ok=True)
    target = np.load(os.path.join(config.dlatents_dir, '1_01.npy'))
 
    #move_and_show(generator,target, "age", [-20, -16, -12, -8, 0, 8, 12, 16, 20])
    #move_and_show(generator,target, "race_black", [-40, -32, -24, -16, 0, 16, 24, 32, 40])
    #move_and_show(generator,target, "gender", [-20, -16, -12, -8, 0, 8, 12, 16, 20])
    move_and_show(generator,target, "eyes_open", [-12,-11,-10,-9,-8,-7,-6, -5,-4,-2,-1,0, 1,2,3,4,5,6,7,8,9,10,12,14,16,18,19,20,21,22,23,24])
    #move_and_show(generator,target, "glasses", [-16, -12, -8, -4, 0, 4, 8, 12, 16])
    #move_and_show(generator,target, "smile", [-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8])
 
if __name__ == "__main__":
    main()