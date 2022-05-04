import torch

def content_loss(Y_hat, Y): #计算与原图片的mse
    return torch.square(Y_hat - Y.detach()).mean()


def gram(X):#计算channels之间的协方差矩阵
    num_channels = X.shape[1]
    X = X.reshape((num_channels, -1))
    return torch.matmul(X, X.T) / X.numel()

#用channels之间的协方差矩阵代表风格特征
def style_loss(Y_hat, gram_Y):
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean() #计算当前图片channels之间的协方差矩阵和风格图片channels之间的协方差矩阵的mse


def tv_loss(Y_hat):#图片平滑loss，去噪
    tmp = torch.abs(
        Y_hat[:,:,1:,:] - Y_hat[:,:,:-1,:]
    ).mean() + torch.abs(
        Y_hat[:,:,:,1:] - Y_hat[:,:,:,:-1]
    ).mean()
    return 0.5 * tmp


def compute_loss(X, content_Y_hat, style_Y_hat, content_Y, style_Y):#总loss
    #loss权重
    weight = {
        "content":1,
        "style":1e5,
        "tv":10
    }
    content_l = [
        content_loss(Y_hat, Y) * weight['content'] for Y_hat, Y in zip(content_Y_hat, content_Y)
    ]

    style_l = [
        style_loss(Y_hat, Y) * weight['style'] for Y_hat, Y in zip(style_Y_hat, style_Y)
    ]

    tv_l = tv_loss(X) * weight['tv']

    total_l = sum(content_l + style_l + [tv_l])

    return content_l, style_l, tv_l, total_l
