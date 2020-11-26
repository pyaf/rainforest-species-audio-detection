from .main import *

# run from root directory, with `python -m models.test`
if __name__ == "__main__":
    # model_name = "se_resnext50_32x4d_v2"
    # model_name = "nasnetamobile"
    # model_name = "resnext101_32x4d_v0"
    #model_name = "efficientnet-b5"
    model_name = 'resnet34'
    classes = 1108
    size = 512
    #model = get_model(model_name, classes, "imagenet")
    model = ArcNet()
    image1 = torch.Tensor(
        3, 3, size, size
    )  # BN layers need more than one inputs, running mean and std
    # image = torch.Tensor(1, 3, 112, 112)
    # image = torch.Tensor(1, 3, 96, 96)
    image2 = torch.Tensor(
        3, 3, size, size
    )  # BN layers need more than one inputs, running mean and std

    pdb.set_trace()
    #label = torch.zeros([3, 1108])
    #label[:, 5] = 1
    #label = torch.LongTensor([[100, 102, 101]]) # 3x3
    label = torch.LongTensor([[100], [102], [101]])
    #output = model(image1, image2)
    output = model(image1, label)
    print(output.shape)

