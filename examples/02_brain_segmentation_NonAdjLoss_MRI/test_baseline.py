import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from collections import OrderedDict

# from torchmed.utils.preprocessing import N4BiasFieldCorrection

from architecture import ModSegNet

def preprocess(img):
    
    # Normalize
    img_max = img.max()
    img_min = img.min()
    return (img - img_min)/(img_max - img_min + 1e-7)

    # Center and Standardize
    mean = img.mean()
    std = img.std() + 1e-7
    return (img - mean) / std

# # original saved file with DataParallel
# state_dict = torch.load('myfile.pth.tar')
# # create new OrderedDict that does not contain `module.`
# from collections import OrderedDict
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = k[7:] # remove `module.`
#     new_state_dict[name] = v
# # load params
# model.load_state_dict(new_state_dict)

def N4BiasFieldCorrection(image, destination, nb_iteration=50):
    inputImage = sitk.ReadImage(image)

    inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    # corrector.SetMaximumNumberOfIterations(nb_iteration)

    output = corrector.Execute(inputImage)
    sitk.WriteImage(output, destination)


model = ModSegNet(num_classes=136,
                    n_init_features=7).cuda()

torch.backends.cudnn.benchmark = True
checkpoint = torch.load('output/checkpoint_105.pth.tar')
new_state_dict = OrderedDict()
for k, v in checkpoint['state_dict'].items():
    name = "module." + k
    new_state_dict[name] = v 
try:
    model.load_state_dict(new_state_dict)
except RuntimeError as e:
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(new_state_dict)
    model = model.module

# brain = N4BiasFieldCorrection('1039_3.nii.gz', './brain.nii.gz')
brain = sitk.GetArrayFromImage(sitk.ReadImage('brain.nii.gz'))
# brain = preprocess(brain)

input = np.zeros((1,7,184,184))
for i in range(7):
    # slice = np.load(f'../../../TF2_Keras_brainseg/data2d/training-images-coronal-full/1000_3_{i+100}.npy')
    # slice = preprocess(slice)
    slice = brain[100+i,:,:]
    slice = preprocess(slice)
    plt.imshow(slice)
    plt.show()
    slice = slice[:184, :184]
    input[0,i,:,:] = slice
input = torch.Tensor(input)

output = model(input.cuda())
_, predicted = output.data.max(1)
plt.imshow(predicted[0].cpu().numpy())
plt.show()