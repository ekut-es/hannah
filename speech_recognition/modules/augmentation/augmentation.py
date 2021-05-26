import os
import shutil
import random
import subprocess
import threading

import xml.etree.ElementTree as ET

from speech_recognition.datasets.base import DatasetType

from speech_recognition.datasets.Kitti import Kitti


class XmlAugmentationParser:
    @staticmethod
    def parse(conf, img, path):
        random.seed()
        augmentation = random.choice(conf["augmentations"])

        if "rain" in augmentation:
            XmlAugmentationParser.__parseRain(
                dict((key, a[key]) for a in conf["rain_drops"] for key in a), img, path
            )
        elif "snow" in augmentation:
            XmlAugmentationParser.__parseSnow(
                dict((key, a[key]) for a in conf["snow"] for key in a), img, path
            )
        elif "fog" in augmentation:
            XmlAugmentationParser.__parseFog(
                dict((key, a[key]) for a in conf["fog"] for key in a), img, path
            )

    @staticmethod
    def __parseRain(conf, img, path):
        tree = ET.parse(path + "/augmentation/rain_drops.xml")
        root = tree.getroot()

        for params in root.iter("ParameterList"):
            for param in params:
                description = param.attrib["Description"]
                if description == "angle of rain streaks [deg]":
                    value = conf["angle_rain_streaks"].split(",")
                    param.attrib["Value"] = str(
                        random.uniform(float(value[0]), float(value[1]))
                    )
                elif description == "Brightness factor":
                    value = conf["brightness"].split(",")
                    param.attrib["Value"] = str(
                        random.uniform(float(value[0]), float(value[1]))
                    )
                elif description == "Number of Drops":
                    value = conf["number_drops"].split(",")
                    param.attrib["Value"] = str(
                        random.randint(int(value[0]), int(value[1]))
                    )
                elif description == "Rain Rate [mm/h]":
                    value = conf["rain_rate"].split(",")
                    param.attrib["Value"] = str(
                        random.uniform(float(value[0]), float(value[1]))
                    )
                elif description == "Mean Drop Radius [m]":
                    value = conf["drop_radius"].split(",")
                    param.attrib["Value"] = str(
                        random.uniform(float(value[0]), float(value[1]))
                    )
                elif description == "Output filename":
                    param.attrib["Value"] = img[:-4]

        tree.write(path + "/augmentation/augment.xml")

    @staticmethod
    def __parseSnow(conf, img, path):
        tree = ET.parse(path + "/augmentation/snow.xml")
        root = tree.getroot()

        for params in root.iter("ParameterList"):
            for param in params:
                description = param.attrib["Description"]
                if description == "Snow Fall Rate [mm/h]":
                    value = conf["snowfall_rate"].split(",")
                    param.attrib["Value"] = str(
                        random.uniform(float(value[0]), float(value[1]))
                    )
                elif description == "Car Speed [m/s]":
                    value = conf["car_speed_ms"].split(",")
                    param.attrib["Value"] = str(
                        random.uniform(float(value[0]), float(value[1]))
                    )
                elif description == "Crosswind Speed [m/s]":
                    value = conf["car_speed_ms"].split(",")
                    param.attrib["Value"] = str(
                        random.uniform(float(value[0]), float(value[1]))
                    )
                elif description == "Draw Fog":
                    value = random.choice(conf["draw_fog"])
                    param.attrib["Value"] = value
                elif description == "Output filename":
                    param.attrib["Value"] = img[:-4]

        tree.write(path + "/augmentation/augment.xml")

    @staticmethod
    def __parseFog(conf, img, path):
        tree = ET.parse(path + "/augmentation/fog.xml")
        root = tree.getroot()

        for params in root.iter("ParameterList"):
            for param in params:
                description = param.attrib["Description"]
                if description == "Fog Density [1/um^3]":
                    value = conf["fog_density"].split(",")
                    param.attrib["Value"] = str(
                        random.uniform(float(value[0]), float(value[1]))
                    )
                elif description == "Fog Sphere Diameter [um]":
                    value = conf["fog_sphere"].split(",")
                    param.attrib["Value"] = str(
                        random.uniform(float(value[0]), float(value[1]))
                    )
                elif description == "Output filename":
                    param.attrib["Value"] = img[:-4]

        tree.write(path + "/augmentation/augment.xml")


class AugmentationThread:
    def __init__(self):
        self.imgs_total = 0
        self.augmented_imgs = 0
        # self.stop = True
        # self.running = False

    def call_augment(self, conf, img, kitti_dir):
        XmlAugmentationParser.parse(conf, img, kitti_dir)
        subprocess.call(kitti_dir + "/augmentation/perform_augmentation.sh")

    def augment(self, conf, kitti, pct):
        # self.running = True
        # self.stop = False
        kitti.aug_files = list()

        for img in kitti.img_files:
            """if self.stop:
            break"""

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
        # self.stop = True
        # self.running = False

    """def clear(self):
        self.stop = True

        while self.running:
            continue
"""


class Augmentation:
    def __init__(self, augmentation: list()):
        self.aug_thread = AugmentationThread()
        self.conf = dict((key, a[key]) for a in augmentation for key in a)
        self.pct = self.conf["augmented_pct"] if "augmented_pct" in self.conf else 0

    def augment(self, kitti: Kitti):
        kitti.aug_files = list()

        if self.pct != 0:
            """self.aug_thread.clear()
            th = threading.Thread(
                target=self.aug_thread.augment,
                args=(
                    self.conf,
                    kitti,
                    self.pct if kitti.set_type == DatasetType.TRAIN else 50,
                ),
            )
            th.start()"""
            self.aug_thread.augment(
                self.conf,
                kitti,
                self.pct if kitti.set_type == DatasetType.TRAIN else 100,
            )

    def getPctAugmented(self):
        return (
            self.aug_thread.augmented_imgs / self.aug_thread.imgs_total
            if self.aug_thread.imgs_total != 0
            else 0
        )
