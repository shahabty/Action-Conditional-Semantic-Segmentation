import os
import numpy as np
import json
from PIL import Image
from torch.utils import data
import random
import torchvision.transforms.functional as TF

palette = [0,0,0,70, 70, 70,190, 153, 153,72,0,90,220,20,60,153,153,153,157,234,50,128,64,128,244,35,232,107,142,35,0,0,255,102,102,156,220,220,0]


zero_pad = 256 * 3 - len(palette)

for i in range(zero_pad):
    palette.append(0)

class carla(data.Dataset):
    '''
    :param int required_length: the required count of items to generate for training (this value is not used in val mode)
    :param int frames_for_each_item: how many frames for each individual item should be returned (including the ground truth as well)
    :param int skip_within_item: the amount of frames skipped within output frames of a single dataset entry
    :param int skip_between_items: the amount of frames skipped between different entries of dataset (this value is not used in val mode)
    :param list val_episodes_list: the episode names of the validation sets
    '''
    def __init__(self,mode = 'train',root = '/mnt/creeper/grad/nabaviss/carla/raw',input_width = 512,input_height = 256,input_transform = None,target_transform = None,required_length = 2000,frames_for_each_item = 5, skip_within_item = 5, skip_between_items = 5, val_episodes_list = ['episode_00001']):
        
        self.mode = mode
        self.root = root
        self.input_width = input_width
        self.input_height = input_height
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.frames_for_each_item = frames_for_each_item
        self.skip_within_item = skip_within_item
        
        if mode == 'train':
            self.training = True
            self.root += '/town1'
            self.skip_between_items = skip_between_items
            self.required_length = required_length

        if mode == 'val':
            self.training = False
            self.root +='/town2'
            self.val_episodes_list = val_episodes_list
            self.skip_between_items = self.skip_within_item

        self.items = self.make_dataset()
        
        if self.training:
            total_items = self.__get_total_frames_across_categories__()
            assert(total_items > (self.required_length * self.__get_each_item_length__()))

        self.id_to_trainid = {0:0, 1: 1,2: 2, 3: 3, 4: 4,5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11,12:12}

    def colorize_mask(mask):
        # mask: numpy array of the mask
        new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
        new_mask.putpalette(palette)
        return TF.to_tensor(new_mask.convert('RGB'))

    def make_dataset(self):

        if self.mode == None:
            return 0

        frames = {}

        categories = sorted(os.listdir(self.root))

        for c in categories:
            if c.endswith('.json'):
                continue
            
            if not self.training and c not in self.val_episodes_list:
                continue

            frames[c] = []

            for x in sorted(os.listdir(os.path.join(self.root,c))):
                if x.startswith('CentralSemanticSeg_'):
                    frames[c].append(os.path.join(self.root, c, x))

        return frames

    def __getitem__(self, index):        
        
        input_frames = []        

        if (self.training):

            rand_category = self.__get_random_category__()
            index = random.randint(0, len(self.items[rand_category]) - self.__get_each_item_length__())

            for i in range(self.frames_for_each_item):
                input_frames.append(self.items[rand_category][index + (i * self.skip_within_item)])

        else:
            
            for i in range(self.frames_for_each_item):
                input_frames.append(self.items[self.__get_category_of_index__(index)][index + (i * self.skip_within_item)])
        
        gts = None
        data_future = []
        speed_future = []
        steer_range = 70.0

        result_frames = []

        for i, inp in enumerate(input_frames):

            if i == self.frames_for_each_item - 1:
                gts = np.moveaxis(np.array(Image.open(inp).convert('RGB'),dtype = np.float32),-1,0)[0,:,:]

            else:
                result_frames.append(np.moveaxis(np.array(Image.open(inp).convert('RGB'), dtype = np.float32),-1,0)[0,:,:])

            with open(inp.replace('CentralSemanticSeg_','measurements_').replace('png','json')) as v:

                data = json.load(v)
                steer = data['steer']*steer_range + 70
                
                if data['stop_vehicle'] == 1:
                    speed = 0
                else:
                    speed = 1
                
                if i == self.frames_for_each_item - 2:
                    data_future.append(steer)
                    speed_future.append(speed)
       
        result_frames_stacked = np.stack(result_frames, 0)
        
        return np.array(data_future),np.array(speed_future),result_frames_stacked,gts

    def __len__(self):
        
        if self.training:
            return self.required_length
        
        total_items = 0
    
        for frames in self.items.values():

            total_items += len(frames) // self.__get_each_item_length__()

        return total_items

    def __get_random_category__(self):
        
        rand_category = random.randint(0,self.__get_total_frames_across_categories__())
        
        total_frames_counted = 0

        for category, frames in self.items.items():

            total_frames_counted += len(frames)

            if rand_category <= total_frames_counted:
                return category

    def __get_category_of_index__(self, index):

        total_frames_counted = 0

        for category, frames in self.items.items():
            
            total_frames_counted += len(frames) // self.__get_each_item_length__()

            if index <= total_frames_counted:
                return category        
        
    def __get_total_frames_across_categories__(self):
    
        total_frames = 0

        for frames in self.items.values():
            total_frames += len(frames)

        return total_frames

    def __get_each_item_length__(self):

        return ((self.frames_for_each_item - 1) * self.skip_within_item + self.skip_between_items + 1)


if __name__ == "__main__":
    
    dataset = carla(mode='val',val_episodes_list=['episode_00001'])
    d = dataset.__getitem__(30)

    print(f'data_future = {d[0]}')
    print(f'speed_future = {d[1]}')
    print(f'frames.shape = {d[2].shape}')
    print(f'gts.shape = {d[3].shape}')
