import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt # Optional: for displaying images in a notebook

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

import copy
import os

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Desired size of the output image
# For faster processing, start with a smaller size like 256 or 512
# For higher quality final NST output (before upscaling), you might go up to 1024 or more
# if your GPU can handle it. This is the size of the *shorter* edge if images are not square.
imsize = 512  # Use a smaller size if on CPU

# --- Image Loading and Preprocessing ---
loader = transforms.Compose([
    transforms.Resize(imsize),  # Scale imported image
    transforms.CenterCrop(imsize), # Crop to ensure square if not already, or just resize if one dim is imsize
    transforms.ToTensor()])  # Transform it into a torch tensor

def image_loader(image_name):
    image = Image.open(image_name).convert('RGB') # Ensure image is RGB
    # Fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# --- Model Definition ---
cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

# VGG19 normalization mean and std (from ImageNet)
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# Create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # broadcast to [N x C x H x W].
        # N is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

# --- Loss Functions ---
class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # We 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # Reshape F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # Compute the gram product

    # We 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

# --- Building the Model with Losses ---
# Desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    # Normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # Just in order to have an iterable access to or list of content/syle losses
    content_losses = []
    style_losses = []

    # Assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # Increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # Add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # Add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # Now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

# --- Optimizer ---
def get_input_optimizer(input_img):
    # This line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

# --- Style Transfer Loop ---
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)

    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # Correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # A last correction...
    input_img.data.clamp_(0, 1)

    return input_img


# --- Main Execution ---
if __name__ == '__main__':
    # --- !!! IMPORTANT: DEFINE YOUR IMAGE PATHS HERE !!! ---
    
    # Choose one style image from your 'high_quality_images' folder
    # Example: style_img_path = r"C:\Users\THILAK R\OneDrive\Desktop\high_quality_images\Vincent_van_Gogh-Zwei_Bäuerinnen_bei....jpg"
    # Make sure to replace with an actual filename from your folder.
    # Using raw string (r"...") or forward slashes is good for Windows paths.
    style_img_path = r"C:\Users\THILAK R\OneDrive\Desktop\high_quality_images\Vincent_van_Gogh-Zwei_Bäuerinnen_bei_der_Kartoffelernte-03986.jpg" # <<< REPLACE THIS

    # Choose one content image from your 'image_content' folder (COCO images)
    # Example: content_img_path = r"C:\Users\THILAK R\OneDrive\Desktop\image_content\000000000009.jpg"
    content_img_path = r"C:\Users\THILAK R\OneDrive\Desktop\image_content\000000293804.jpg"
    # Define where you want to save the output and what to call it
    output_dir = r"C:\Users\THILAK R\OneDrive\Desktop\stylized_outputs"
    os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist
    
    # Construct a unique output name, perhaps incorporating style and content image names
    style_name = os.path.splitext(os.path.basename(style_img_path))[0]
    content_name = os.path.splitext(os.path.basename(content_img_path))[0]
    output_img_name = os.path.join(output_dir, f"{content_name}_styled_by_{style_name}.jpg")

    # --- Check if image paths are valid before proceeding ---
    if not os.path.exists(style_img_path):
        print(f"ERROR: Style image not found at {style_img_path}")
        exit()
    if not os.path.exists(content_img_path):
        print(f"ERROR: Content image not found at {content_img_path}")
        exit()

    style_img = image_loader(style_img_path)
    content_img = image_loader(content_img_path)

    assert style_img.size() == content_img.size(), \
        "We need to import style and content images of the same size for this tutorial version"

    # Start with a copy of the content image, or random noise
    input_img = content_img.clone()
    # if you want to use white noise instead:
    # input_img = torch.randn(content_img.data.size(), device=device)

    # --- Run the Style Transfer ---
    print(f"Processing content image: {content_img_path}")
    print(f"With style image: {style_img_path}")
    
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img, num_steps=500) # Start with 300 steps

    # --- Save the Output ---
    save_image(output, output_img_name)
    print(f"Stylized image saved to: {output_img_name}")

    # Optional: Display images if you are in an environment like Jupyter Notebook
    # plt.figure()
    # unloader = transforms.ToPILImage() # Reconvert into PIL image
    # plt.imshow(unloader(output.cpu().clone().squeeze(0)))
    # plt.title('Output Image')
    # plt.ioff()
    # plt.show()