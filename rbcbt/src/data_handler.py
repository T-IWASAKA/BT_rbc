# -*- coding: utf-8 -*-
"""
Created 2025

data handler

@author: iwasaka
"""

## import ##

def rbc_dataset(tif_file, patch_size=1024, goal=10000, splitn=1, check_ditect=True, split_sample=False):
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

    if split_sample:
        random.shuffle(total_image)
        my_datasets = [SmearDataset_RBC(i) for i in np.array_split(total_image, splitn)]
    else:
        my_datasets = [SmearDataset_RBC(total_image)]

    if n == 1:
        print("This file contains corrupted regions.")

    #print("Number of  datasets :", len(my_datasets))
    #print("Number of images per dataset :", len(my_datasets[0]))
    print("\n")

    return my_datasets

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1 #　初期値random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        return img


class RandomRotate(object):
    """Implementation of random rotation.
    Randomly rotates an input image by a fixed angle. By default, we rotate
    the image by 90 degrees with a probability of 50%.
    This augmentation can be very useful for rotation invariant images such as
    in medical imaging or satellite imaginary.
    Attributes:
        prob:
            Probability with which image is rotated.
        angle:
            Angle by which the image is rotated. We recommend multiples of 90
            to prevent rasterization artifacts. If you pick numbers like
            90, 180, 270 the tensor will be rotated without introducing 
            any artifacts.
    
    """

    def __init__(self, prob: float = 0.5, angle: Union[int, list, tuple] = None):
        self.prob = prob
        self.angle = [90] if angle is None else list(angle) 

    def __call__(self, sample):
        """Rotates the images with a given probability.
        Args:
            sample:
                PIL image which will be rotated.
        
        Returns:
            Rotated image or original image.
        """
        prob = np.random.random_sample()
        selected_angle = int(np.random.choice(self.angle))
        if prob < self.prob:
            sample =  transforms.functional.rotate(sample, selected_angle)
        return sample


def random_rotation_transform(
    rr_prob: float = 0.5,
    rr_degrees: Union[None, float, Tuple[float, float]] = 90,
    ) -> Union[RandomRotate, transforms.RandomApply]:
    if rr_degrees == 90:
        # Random rotation by 90 degrees.
        return RandomRotate(prob=rr_prob, angle=[90, 180, 270])
    else:
        # Random rotation with random angle defined by rr_degrees.
        return transforms.RandomApply([transforms.RandomRotation(degrees=rr_degrees)], p=rr_prob)


class SSLTransform:
    def __init__(self, transform=None, transform_prime=None, crop_size=None) -> None:
        """
        transform for self-supervised learning

        Parameters
        ----------
        transform: torchvision.transforms
            transform for the original image

        transform_prime: torchvision.transforms
            transform to be applied to the second
        
        """
        if crop_size is None:
            crop_size = 32
        else:
            pass
        if transform is None:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(crop_size, scale=(0.95, 1.0), interpolation=Image.BICUBIC),#
                transforms.RandomHorizontalFlip(p=0.5),#
                random_rotation_transform(rr_prob=1., rr_degrees=[0,180]),## 初期値rr_degrees=[0,180], 90の倍数で回したい場合はrr_degrees=90
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], # default brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)]
                    p=0.8
                    ),#
                transforms.RandomGrayscale(p=0.2),# default 0.2
                GaussianBlur(p=0.2),#
                #Solarization(p=0),#
                transforms.ToTensor(),#
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))#
            ])
        else:
            self.transform = transform
        if transform_prime is None:
            self.transform_prime = transforms.Compose([
                transforms.RandomResizedCrop(crop_size, scale=(0.95, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], # default brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)]
                    p=0.9
                    ),
                transforms.RandomGrayscale(p=0.2),# default 0.2
                GaussianBlur(p=0.2),# default 0.5
                #Solarization(p=0),# default 0.2
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))# default (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            ])
        else:
            self.transform_prime = transform_prime
        

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2
    


class Dataset_BT(torch.utils.data.Dataset):
    """ to create my dataset """
    def __init__(self, mydataset, transform):
        if transform is None:
            raise ValueError('!! Give transform !!')
        self.transform = [transform]
        self.input = [mydataset[0][i][0] for i in range(len(mydataset[0]))]
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
