# %%
import torch
import matplotlib.pyplot as plt
from PIL import Image
from styleTransfer import *

# %%
style_path = "./style_pic.jpeg"
content_path = "./pic2.jpg"

# %%
style_pic = Image.open(style_path)
content_pic = Image.open(content_path)

# %%
style_layer = [0, 5, 10, 19, 28]
content_layer = [25]
device = torch.device("cuda:0")

# %%
transfer = styleTransfer(content_pic, style_pic, style_layer, content_layer, device)

# %%
transfer.init(content_pic, style_pic, (600, 1000))
transfer.train(300, 0.01, plot_loss=True)
result = transfer.getResult()
plt.imshow(result)

result.save("./result2.png")