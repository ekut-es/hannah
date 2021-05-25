import os
import shutil
import random
import subprocess
import threading

import xml.etree.ElementTree as ET

from speech_recognition.datasets.base import DatasetType

from speech_recognition.datasets.Kitti import Kitti


# TODO parse rain, fog and snow (Params in conf, daraus zufällig werte)
class XmlAugmentationParser:
    @staticmethod
    def parse(conf, img, path):
        random.seed()
        augmentation = random.choice(conf["augmentations"])

        if "rain" in augmentation:
            XmlAugmentationParser.__parseRain(conf, img, path)

    @staticmethod
    def __parseRain(conf, img, path):
        tree = ET.parse(path + "/augmentation/rain_drops.xml")
        root = tree.getroot()

        for params in root.iter("ParameterList"):
            for param in params:
                if param.attrib["Description"] == "Output filename":
                    param.attrib["Value"] = img[:-4]

        tree.write(path + "/augmentation/augment.xml")


class AugmentationThread:
    def __init__(self):
        self.imgs_total = 0
        self.augmented_imgs = 0

    def call_augment(self, conf, img, kitti_dir):
        XmlAugmentationParser.parse(conf, img, kitti_dir)
        subprocess.call(kitti_dir + "/augmentation/perform_augmentation.sh")

    def augment(self, conf, kitti, pct):
        kitti.aug_files = list()

        for img in kitti.img_files:
            random.seed()
            rand = random.randrange(0, 100)

            if rand < pct:
                txt = open(kitti.kitti_dir + "/augmentation/to_augment.txt", "w")
                kitti.aug_files.append(img[:-4])
                txt.write(img[:-4] + "\n")
                txt.close()
                self.call_augment(conf, img, kitti.kitti_dir)
                self.augmented_imgs += 1
            self.imgs_total += 1


class Augmentation:
    def __init__(self, augmentation: list()):
        self.aug_thread = AugmentationThread()
        self.conf = dict((key, a[key]) for a in augmentation for key in a)
        self.pct = self.conf["augmented_pct"] if "augmented_pct" in self.conf else 0

    def augment(self, kitti: Kitti):
        kitti.aug_files = list()

        if self.pct != 0:
            th = threading.Thread(
                target=self.aug_thread.augment,
                args=(
                    self.conf,
                    kitti,
                    self.pct if kitti.set_type == DatasetType.TRAIN else 50,
                ),
            )
            th.start()

    def getPctAugmented(self):
        return (
            self.aug_thread.augmented_imgs / self.aug_thread.imgs_total
            if self.aug_thread.imgs_total != 0
            else 0
        )
