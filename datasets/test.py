
if __name__ == "__main__":
    ''' doesn't work, gotta set seeds at function level
    seed = 69
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    '''
    import os
    os.environ['CUDA_VISIBLE_DEVICES']=""
    import torchvision
    torchvision.set_image_backend('accimage')

    from utils.extras import *
    from .dataset_factory import *
    import time
    start = time.time()
    phase = "train"
    args = get_parser()
    cfg = load_cfg(args)
    cfg["num_workers"] = 10
    cfg["batch_size"]["train"] = 2
    cfg["batch_size"]["val"] = 2

    df = pd.read_csv(str(cfg.home / cfg.df_path))
    dataloader = provider(df, phase, cfg)
    ''' train val set sanctity
    #pdb.set_trace()
    tdf = dataloader.dataset.df
    phase = "val"
    dataloader = provider(phase, cfg)
    vdf = dataloader.dataset.df
    print(len([x for x in tdf.id_code.tolist() if x in vdf.id_code.tolist()]))
    exit()
    '''
    total_labels = []
    total_len = len(dataloader)
    from collections import defaultdict
    fnames_dict = defaultdict(int)
    for idx, batch in enumerate(dataloader):
        fnames, images, labels = batch
        #import pdb; pdb.set_trace()
        #class_indices = torch.max(labels, 1)[1].tolist()

        #labels = (torch.sum(labels, 1) - 1).numpy().astype('uint8')
        for fname in fnames:
            fnames_dict[fname] += 1

        print("%d/%d" % (idx, total_len), images.shape, labels.shape)
        #total_labels.extend(class_indices)
        #pdb.set_trace()
    print(np.unique(total_labels, return_counts=True))
    diff = time.time() - start
    print('Time taken: %02d:%02d' % (diff//60, diff % 60))

    print(np.unique(list(fnames_dict.values()), return_counts=True))
    #pdb.set_trace()


"""
Footnotes:

https://github.com/btgraham/SparseConvNet/tree/kaggle_Diabetic_Retinopathy_competition

[1] CrossEntropyLoss doesn't expect inputs to be one-hot, but indices
[2] .value_counts() returns in descending order of counts (not sorted by class numbers :)
[3]: bad_indices are those which have conflicting diagnosises, duplicates are those which have same duplicates, we shouldn't let them split in train and val set, gotta maintain the sanctity of val set
[4]: used when the dataframe include external data and we want to sample limited number of those
[5]: as replace=False,  total samples can be a finite number so that those many number of classes exist in the dataset, and as the count_dist is approx, not normalized to 1, 7800 is optimum, totaling to ~8100 samples

[6]: albumentations.Normalize will divide by 255, subtract mean and divide by std. output dtype = float32. ToTensor converts to torch tensor and divides by 255 if input dtype is uint8.
[7]: indices of hard examples, evaluated using 0.81 scoring model.
[10]: messidor df append will throw err when doing hard ex sampling.
[11]: using current comp's data as val set in old data training.
[12]: messidor's class 3 is class 3 and class 4 combined.
"""
