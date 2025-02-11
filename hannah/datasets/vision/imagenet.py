import logging
import os
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch.utils.data as data
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import verify_str_arg

from .base import TorchvisionDatasetBase

CLASS_NAMES = [
    "African elephant, Loxodonta africana",
    "American alligator, Alligator mississipiensis",
    "American lobster, Northern lobster, Maine lobster, Homarus americanus",
    "Arabian camel, dromedary, Camelus dromedarius",
    "CD player",
    "Chihuahua",
    "Christmas stocking",
    "Egyptian cat",
    "European fire salamander, Salamandra salamandra",
    "German shepherd, German shepherd dog, German police dog, alsatian",
    "Labrador retriever",
    "Persian cat",
    "Yorkshire terrier",
    "abacus",
    "academic gown, academic robe, judge's robe",
    "acorn",
    "albatross, mollymawk",
    "alp",
    "altar",
    "apron",
    "baboon",
    "backpack, back pack, knapsack, packsack, rucksack, haversack",
    "banana",
    "bannister, banister, balustrade, balusters, handrail",
    "barbershop",
    "barn",
    "barrel, cask",
    "basketball",
    "bathtub, bathing tub, bath, tub",
    "beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon",
    "beacon, lighthouse, beacon light, pharos",
    "beaker",
    "bee",
    "beer bottle",
    "bell pepper",
    "bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis",
    "bikini, two-piece",
    "binoculars, field glasses, opera glasses",
    "birdhouse",
    "bison",
    "black stork, Ciconia nigra",
    "black widow, Latrodectus mactans",
    "boa constrictor, Constrictor constrictor",
    "bow tie, bow-tie, bowtie",
    "brain coral",
    "brass, memorial tablet, plaque",
    "broom",
    "brown bear, bruin, Ursus arctos",
    "bucket, pail",
    "bullet train, bullet",
    "bullfrog, Rana catesbeiana",
    "butcher shop, meat market",
    "candle, taper, wax light",
    "cannon",
    "cardigan",
    "cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM",
    "cauliflower",
    "centipede",
    "chain",
    "chest",
    "chimpanzee, chimp, Pan troglodytes",
    "cliff dwelling",
    "cliff, drop, drop-off",
    "cockroach, roach",
    "comic book",
    "computer keyboard, keypad",
    "confectionery, confectionary, candy store",
    "convertible",
    "coral reef",
    "cougar, puma, catamount, mountain lion, painter, panther, Felis concolor",
    "crane",
    "dam, dike, dyke",
    "desk",
    "dining table, board",
    "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk",
    "drumstick",
    "dugong, Dugong dugon",
    "dumbbell",
    "espresso",
    "flagpole, flagstaff",
    "fly",
    "fountain",
    "freight car",
    "frying pan, frypan, skillet",
    "fur coat",
    "gasmask, respirator, gas helmet",
    "gazelle",
    "go-kart",
    "golden retriever",
    "goldfish, Carassius auratus",
    "gondola",
    "goose",
    "grasshopper, hopper",
    "guacamole",
    "guinea pig, Cavia cobaya",
    "hog, pig, grunter, squealer, Sus scrofa",
    "hourglass",
    "iPod",
    "ice cream, icecream",
    "ice lolly, lolly, lollipop, popsicle",
    "jellyfish",
    "jinrikisha, ricksha, rickshaw",
    "kimono",
    "king penguin, Aptenodytes patagonica",
    "koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus",
    "ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle",
    "lakeside, lakeshore",
    "lampshade, lamp shade",
    "lawn mower, mower",
    "lemon",
    "lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens",
    "lifeboat",
    "limousine, limo",
    "lion, king of beasts, Panthera leo",
    "magnetic compass",
    "mantis, mantid",
    "mashed potato",
    "maypole",
    "meat loaf, meatloaf",
    "military uniform",
    "miniskirt, mini",
    "monarch, monarch butterfly, milkweed butterfly, Danaus plexippus",
    "moving van",
    "mushroom",
    "nail",
    "neck brace",
    "obelisk",
    "oboe, hautboy, hautbois",
    "orange",
    "orangutan, orang, orangutang, Pongo pygmaeus",
    "organ, pipe organ",
    "ox",
    "parking meter",
    "pay-phone, pay-station",
    "picket fence, paling",
    "pill bottle",
    "pizza, pizza pie",
    "plate",
    "plunger, plumber's helper",
    "pole",
    "police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria",
    "pomegranate",
    "poncho",
    "pop bottle, soda bottle",
    "potpie",
    "potter's wheel",
    "pretzel",
    "projectile, missile",
    "punching bag, punch bag, punching ball, punchball",
    "reel",
    "refrigerator, icebox",
    "remote control, remote",
    "rocking chair, rocker",
    "rugby ball",
    "sandal",
    "school bus",
    "scoreboard",
    "scorpion",
    "sea cucumber, holothurian",
    "sea slug, nudibranch",
    "seashore, coast, seacoast, sea-coast",
    "sewing machine",
    "slug",
    "snail",
    "snorkel",
    "sock",
    "sombrero",
    "space heater",
    "spider web, spider's web",
    "spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish",
    "sports car, sport car",
    "standard poodle",
    "steel arch bridge",
    "stopwatch, stop watch",
    "sulphur butterfly, sulfur butterfly",
    "sunglasses, dark glasses, shades",
    "suspension bridge",
    "swimming trunks, bathing trunks",
    "syringe",
    "tabby, tabby cat",
    "tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui",
    "tarantula",
    "teapot",
    "teddy, teddy bear",
    "thatch, thatched roof",
    "torch",
    "tractor",
    "trilobite",
    "triumphal arch",
    "trolleybus, trolley coach, trackless trolley",
    "turnstile",
    "umbrella",
    "vestment",
    "viaduct",
    "volleyball",
    "walking stick, walkingstick, stick insect",
    "water jug",
    "water tower",
    "wok",
    "wooden spoon"
]

logger = logging.getLogger(__name__)


class TinyImageNet(TorchvisionDatasetBase):
    @classmethod
    def prepare(cls, config):
        data_folder = config.data_folder
        root_folder = os.path.join(data_folder, "tiny-imagenet-200")

    @classmethod
    def splits(cls, config):
        data_folder = config.data_folder
        root_folder = os.path.join(data_folder, "tiny-imagenet-200")

        test_set = TinyImageNetDataset(
            root_folder, split="val"
        )
        train_val_set = TinyImageNetDataset(
            root_folder, split="train"
        )
        train_val_len = len(train_val_set)

        split_sizes = [
            int(train_val_len * (1.0 - config.val_percent)),
            int(train_val_len * config.val_percent),
        ]
        train_set, val_set = data.random_split(train_val_set, split_sizes)

        return (
            cls(config, train_set),
            cls(config, val_set),
            cls(config, test_set),
        )
    
    @property
    def std(self):
        return (0.229, 0.224, 0.225)

    @property
    def mean(self):
        return (0.485, 0.456, 0.406)

    @property
    def class_names(self):
        return CLASS_NAMES


class TinyImageNetDataset(VisionDataset):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    # The downloaded dataset contains three folders: train, val, test. The test folder doesn't have labels
    # Here, the imgs in val folder are used to create the test_set. 

    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = verify_str_arg(split, "split", ("train", "val"))
        # if download:
        #     self.download()

        pickle_file_path = os.path.join(root, f"{self.split}.pkl")
        # prepare train and test pickle files
        if not os.path.isfile(pickle_file_path):
            if self.split == "train":
                print(f"No train.pkl detected. Preparing pickle file...")
                self._prepare_train_pkl()
            elif self.split == "val":
                print(f"No val.pkl detected. Preparing pickle file...")
                self._prepare_val_pkl()

        # load the pickled numpy arrays
        self.data: Any = []
        self.targets = []
        with open(pickle_file_path, "rb") as f:
            entry = pickle.load(f, encoding="latin1")
            self.data.append(entry["data"])
            self.targets.extend(entry["labels"])

        self.data = np.vstack(self.data)  # [num_samples, CHW]
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _prepare_train_pkl(self):
        train_folder = os.path.join(self.root, "train")
        assert os.path.isdir(train_folder)

        class_map = pd.read_csv(
            f"{self.root}/words.txt", sep="\t", index_col=0, names=["label"]
        )
        pickle_data = {"data": [], "labels": []}
        for root, dirs, files in os.walk(train_folder):
            if root.find("images") != -1:
                for name in files:
                    img_path = os.path.join(root, name)
                    img_array = self._get_np_img(img_path)

                    class_id = os.path.split(os.path.dirname(root))[1]
                    class_name = class_map.loc[class_id].item()
                    target = CLASS_NAMES.index(class_name)

                    pickle_data["data"].append(img_array)
                    pickle_data["labels"].append(target)

        save_path = os.path.join(self.root, "train.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(pickle_data, f)

    def _prepare_val_pkl(self):
        val_folder = os.path.join(self.root, "val")
        assert os.path.isdir(val_folder)

        val_annotations = pd.read_csv(
            f'{val_folder}/val_annotations.txt', sep='\t',
            names=['filename', 'class_id', 'x_min', 'y_min', 'x_max', 'y_max']
        )
        class_map = pd.read_csv(
            f"{self.root}/words.txt", sep="\t", index_col=0, names=["label"]
        )
        pickle_data = {"data": [], "labels": []}
        val_img_folder = os.path.join(val_folder, "images")
        for name in os.listdir(val_img_folder):
            img_path = os.path.join(val_img_folder, name)
            img_array = self._get_np_img(img_path)

            class_id = val_annotations.loc[
                val_annotations["filename"] == name
            ]["class_id"].item()
            class_name = class_map.loc[class_id].item()
            target = CLASS_NAMES.index(class_name)

            pickle_data["data"].append(img_array)
            pickle_data["labels"].append(target)

        save_path = os.path.join(self.root, "val.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(pickle_data, f)

    def _get_np_img(self, img_path):
        with open(img_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")  # shape HWC
            img_array = np.asarray(img).transpose((2, 0, 1))  # shape CHW
        return img_array

    def extra_repr(self) -> str:
        split = self.split
        return f"Split: {split}"
