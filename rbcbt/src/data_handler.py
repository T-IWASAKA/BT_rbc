# -*- coding: utf-8 -*-
"""
Created 2025

data handler

@author: iwasaka14
"""

## import ##
from .RBC_loader import Smear_tiff


class SmearDataset_RBC(torch.utils.data.Dataset):
    """
    # 赤血球一個分のサイズの画像をまとめたデータセット
    # BT、Validに関わらず共通

    """
    def __init__(
            self,
            image_lst
        ):
        self.image_lst = image_lst
        
    def __len__(self):
        return len(self.image_lst)
    
    def __getitem__(self, idx):
        image_np = self.image_lst[idx]

        return image_np, idx # とりあえずidxをラベルとして返す
    

class Dataset_SSL(torch.utils.data.Dataset):
    """
    # SSL用にAugをかけた二つの画像をセットにし、まとめたデータセット
    # BT用、Aug用のtransformはimage_aug.pyに記載
    
    """
    def __init__(self, mydataset, transform):
        if transform is None:
            raise ValueError('!! Give transform !!')
        self.transform = [transform]
        if len(mydataset) > 1:
            raise ValueError('!! Add a dataset that you have not SPLIT! !!')
        self.input = [mydataset[0][i][0] for i in range(len(mydataset[0]))] #mydatasetがlist形式のためmydataset[0]であることに注意　
        self.datanum = len(self.input)
        self.transform_totensor = transforms.Compose([
            transforms.ToTensor(),  # [H, W, C] を [C, H, W] に変換
        ])

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        input = Image.fromarray(self.input[idx]) # 上記判定が不要のため
        t = self.transform
        y1, y2 = t[0](input)
        return y1, y2
    

def prep_rbcdata(tif_file, patch_size=1024, goal=10000, check_ditect=True):
    dat_smear = Smear_tiff(tif_file)
    dat_smear.check_dimensions()
    buffer = patch_size/2

    # patchをランダムに取得するための組み合わせ
    loc_pairs = list(itertools.product(range(0, dat_smear.dimensions[0]//patch_size), range(0, dat_smear.dimensions[1]//patch_size)))
    # ランダムにシャッフルして順番にペアを取得
    random.shuffle(loc_pairs)

    total_image = deque()

    # ペアを順番に処理
    n = 0 # errorの判定に使用
    with tqdm(total=goal, desc="total rbc", file=sys.stderr) as pbar:
        for loc_n, xy in enumerate(loc_pairs):
            if loc_n == len(loc_pairs) - 1:
                print("All regions have been searched.")
            x = int(xy[0]*patch_size + buffer)
            y = int(xy[1]*patch_size + buffer)

            isolated_centroids, isolated_areasize = dat_smear.ditect_rbc(patch_size=patch_size, loc=(x, y), rbc_radius=60)
            if isolated_centroids is not None: # Noneを返したときはエラーなので避ける
                rbc_lst = dat_smear.get_rbcimage(isolated_centroids, isolated_areasize, loc=(x, y))
                total_image.extend(rbc_lst)
                pbar.update(len(rbc_lst))
                if len(total_image) > goal:
                    print("The goal has been reached.")
                    if check_ditect:
                        image_array = np.array(dat_smear.get_area(patch_size=patch_size, loc=(x, y)), dtype=np.uint8)
                        plt.scatter(isolated_centroids[:,0],isolated_centroids[:,1],s=50, marker='h',c='orangered')
                        plt.imshow(image_array)#これは縦横 (y, x)
                        plt.show()
                    break
            else:
                dat_smear = Smear_tiff(tif_file)
                pbar.update(0)
                n = 1
                continue


    total_image = list(total_image)
    total_image = total_image[0:goal]

    if check_ditect:
        show_get_img(total_image)

    return total_image


def prep_bgdata(tif_file, patch_size=1600, goal=10000, bg_p=0.01, check_ditect=True):
    dat_smear = Smear_tiff(tif_file)
    dat_smear.check_dimensions()

    # patchをランダムに取得するための組み合わせ
    loc_pairs = list(itertools.product(range(0, dat_smear.dimensions[0]//patch_size), range(0, dat_smear.dimensions[1]//patch_size)))
    # ランダムにシャッフルして順番にペアを取得
    random.shuffle(loc_pairs)

    total_image = deque()

    # ペアを順番に処理
    with tqdm(total=goal, desc="total bg") as pbar:
        for xy in loc_pairs:
            x = int(xy[0]*patch_size)
            y = int(xy[1]*patch_size)
            background_lst = dat_smear.get_background(patch_size=patch_size, loc=(x, y), rbc_size=80, bg_p=bg_p)
            total_image.extend(background_lst)
            pbar.update(len(background_lst))
            if len(total_image) > goal:
                break

    total_image = list(total_image)
    total_image = total_image[0:goal]

    if check_ditect:
        show_get_img(total_image)

    return total_image


def prep_randomdata(tif_file, patch_size=80, goal=10000, check_ditect=True):
    dat_smear = Smear_tiff(tif_file)
    dat_smear.check_dimensions()

    # patchをランダムに取得するための組み合わせ
    loc_pairs = list(itertools.product(range(0, dat_smear.dimensions[0]//patch_size), range(0, dat_smear.dimensions[1]//patch_size)))
    # ランダムにシャッフルして順番にペアを取得
    random.shuffle(loc_pairs)

    total_image = deque()

    # ペアを順番に処理
    with tqdm(total=goal, desc="total img") as pbar:
        for xy in loc_pairs:
            x = int(xy[0]*patch_size)
            y = int(xy[1]*patch_size)
            image_array = np.array(dat_smear.get_area(patch_size=patch_size, loc=(x, y)))[:,:,:3]
            total_image.extend([image_array])
            pbar.update(1)
            if len(total_image) > goal:
                break
    
    total_image = list(total_image)
    total_image = total_image[0:goal]
    if check_ditect:
        show_get_img(total_image)

    return total_image


def prep_dataset(total_image, splitn=1):
    if splitn == 1:
        random.shuffle(total_image)
        my_datasets = [SmearDataset_RBC(i) for i in np.array_split(total_image, splitn)]
    else:
        my_datasets = [SmearDataset_RBC(total_image)]

    #print("Number of  datasets :", len(my_datasets))
    #print("Number of images per dataset :", len(my_datasets[0]))

    return my_datasets


def prep_btdataset(image_path, num_rbc=2000, show_imagedata=True, ssl_transform=None)  -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    prepare dataset using ImageFolder
    
    Parameters
    ----------
    image_path: list
        the path to the image folder

    num_rbc=2000: int
        the number of red blood cells detected per slide

    show_imagedata=True: bool
        Whether the detected images are confirmed or not
    
    ssl_transform=None: a list of ssl transform functions
    
    """
    if type(image_path) == str:
        image_paths = [image_path]
    elif type(image_path) == list:
        image_paths = image_path
    
    # ここでtrainとtestをスライドから分けるようにする
    N = len(image_paths)
    thresh = int(0.8*N) - 1 # 何となくこの数, ここは後から変える！！

    for n, path in enumerate(image_paths):
        print(path)
        total_image =  prep_rbcdata(tif_file, patch_size=1024, goal=num_rbc, check_ditect=show_imagedata)
        my_datasets = prep_dataset(total_image, splitn=1)

        if n == 0:
            train_dataset = mydataset
        elif n <= thresh:
            train_dataset = data.ConcatDataset([train_dataset, mydataset])
        elif n == thresh+1:
            test_dataset = mydataset
        elif n > thresh+1:
            test_dataset = data.ConcatDataset([test_dataset, mydataset])

    print("===============================================================================")
    print("train:test =", str(len(train_dataset)),":", str(len(test_dataset)))

    return train_dataset, test_dataset


def prep_validdataset():





def prep_dataloader(
    dataset, batch_size, shuffle=None, num_workers=2, pin_memory=True
    ) -> torch.utils.data.DataLoader:
    """
    prepare train and test loader
    
    Parameters
    ----------
    dataset: torch.utils.data.Dataset
        prepared Dataset instance
    
    batch_size: int
        the batch size
    
    shuffle: bool
        whether data is shuffled or not

    num_workers: int
        the number of threads or cores for computing
        should be greater than 2 for fast computing
    
    pin_memory: bool
        determines use of memory pinning
        should be True for fast computing
    
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn
        )    
    return loader


def prep_smeardata_bt(
    image_path=None, batch_size:int=0,
    transform=(None, None), ssl_transform=None, 
    shuffle=(True, False),# デフォルトshuffle=(True, False)
    num_workers:int=2, pin_memory:bool=True, 
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    prepare train and test loader from data
    
    Parameters
    ----------
    image_path: str
        the path to the tiff file
            
    batch_size: int
        the batch size

    transform: a tuple of transform functions
        transform functions for training and test, respectively
        each given as a list

    ssl_transform: a list of ssl transform functions
    
    shuffle: (bool, bool)
        indicates shuffling training data and test data, respectively
    
    num_workers: int
        the number of threads or cores for computing
        should be greater than 2 for fast computing
    
    pin_memory: bool
        determines use of memory pinning
        should be True for fast computing

    """
    # 現在は不要だがそのままおいておく
    if transform[0] is None:
        transform = _default_transform()

    # dataset and dataloader preparation
    if ssl_transform is not None:
        train_dataset, test_dataset = prep_btdataset(image_path, ssl_transform=ssl_transform)
        classes = [train_dataset[i][1] for i in range(len(train_dataset))] + [test_dataset[i][1] for i in range(len(test_dataset))]
        train_loader = prep_dataloader(
            train_dataset, batch_size, shuffle[0], num_workers, pin_memory
            )
        test_loader = prep_dataloader(
            test_dataset, batch_size, shuffle[1], num_workers, pin_memory
            )    

    else:
        raise ValueError("!! Give ssl_transform !!")
        
    return train_loader, test_loader, classes














def _worker_init_fn(worker_id):
    """ fix the seed for each worker """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def _default_transform():
    """ return default transforms """
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(
                (32, 32), scale=(0.8, 1.0),
                ratio=(0.75, 1.3333), interpolation=2
            ),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    return train_transform, test_transform

