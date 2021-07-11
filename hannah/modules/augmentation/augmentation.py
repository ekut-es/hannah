import os
import sys
import random
import subprocess
import threading
import numpy as np

from imagecorruptions import corrupt
import xml.etree.ElementTree as ET
from PIL import Image
from hannah.datasets.base import DatasetType
from hannah.modules.augmentation.bordersearch import Parameter, ParameterRange
import scipy.stats as stats

from hannah.datasets.Kitti import Kitti


class XmlAugmentationParser:
    @staticmethod
    def __getImgSize(path, img):
        pil_img = Image.open(path + "/training/image_2/" + img).convert("RGB")
        width, height = pil_img.size
        return (width, height)

    @staticmethod
    def parse(conf, img, kitti):
        random.seed()
        augmentation = random.choices(conf["augmentations"], conf["augmentations_pct"])[
            0
        ]
        subpring = True

        if "rain" in augmentation:
            XmlAugmentationParser.__parseRain(
                dict((key, a[key]) for a in conf["rain_drops"] for key in a), img, kitti
            )
        elif "snow" in augmentation:
            XmlAugmentationParser.__parseSnow(
                dict((key, a[key]) for a in conf["snow"] for key in a), img, kitti
            )
        elif "fog" in augmentation:
            XmlAugmentationParser.__parseFog(
                dict((key, a[key]) for a in conf["fog"] for key in a), img, kitti
            )
        elif "imagecorruptions" in augmentation:
            XmlAugmentationParser.__imagecorruptions(
                dict((key, a[key]) for a in conf["imagecorruptions"] for key in a),
                img,
                kitti,
            )
            subpring = False
        return subpring

    @staticmethod
    def __getDistValue(low, high):
        range = abs(high - low)
        dist = stats.halfnorm()
        sample = sys.maxsize

        while sample > range:
            sample = dist.rvs(size=1).item()

        return low + sample

    @staticmethod
    def __parseRain(conf, img, kitti):
        tree = ET.parse(kitti.kitti_dir + "/augmentation/rain_drops.xml")
        root = tree.getroot()
        size = XmlAugmentationParser.__getImgSize(kitti.kitti_dir, img)

        for params in root.iter("ParameterList"):
            for param in params:
                description = param.attrib["Description"]
                if description == "angle of rain streaks [deg]":
                    value = conf["angle_rain_streaks"]
                    param.attrib["Value"] = str(
                        XmlAugmentationParser.__getDistValue(
                            float(value[0]), float(value[1])
                        )
                    )
                elif description == "Brightness factor":
                    value = conf["brightness"]
                    param.attrib["Value"] = str(
                        XmlAugmentationParser.__getDistValue(
                            float(value[0]), float(value[1])
                        )
                    )
                elif description == "Number of Drops":
                    value = conf["number_drops"]
                    param.attrib["Value"] = str(
                        int(
                            XmlAugmentationParser.__getDistValue(
                                int(value[0]), int(value[1])
                            )
                        )
                    )
                elif description == "Rain Rate [mm/h]":
                    value = conf["rain_rate"]
                    param.attrib["Value"] = str(
                        XmlAugmentationParser.__getDistValue(
                            float(value[0]), float(value[1])
                        )
                    )
                elif description == "Mean Drop Radius [m]":
                    value = conf["drop_radius"]
                    param.attrib["Value"] = str(
                        XmlAugmentationParser.__getDistValue(
                            float(value[0]), float(value[1])
                        )
                    )
                elif description == "Output filename":
                    param.attrib["Value"] = img[:-4]
                elif description == "Pixel width":
                    param.attrib["Value"] = str(size[0])
                elif description == "Pixel height":
                    param.attrib["Value"] = str(size[1])
                elif description == "Output directory":
                    param.attrib["Value"] = kitti.aug_path

        tree.write(kitti.kitti_dir + "/augmentation/augment.xml")

    @staticmethod
    def __parseSnow(conf, img, kitti):
        tree = ET.parse(kitti.kitti_dir + "/augmentation/snow.xml")
        root = tree.getroot()
        size = XmlAugmentationParser.__getImgSize(kitti.kitti_dir, img)

        for params in root.iter("ParameterList"):
            for param in params:
                description = param.attrib["Description"]
                if description == "Snow Fall Rate [mm/h]":
                    value = conf["snowfall_rate"]
                    param.attrib["Value"] = str(
                        XmlAugmentationParser.__getDistValue(
                            float(value[0]), float(value[1])
                        )
                    )
                elif description == "Car Speed [m/s]":
                    value = conf["car_speed_ms"]
                    param.attrib["Value"] = str(
                        XmlAugmentationParser.__getDistValue(
                            float(value[0]), float(value[1])
                        )
                    )
                elif description == "Crosswind Speed [m/s]":
                    value = conf["car_speed_ms"]
                    param.attrib["Value"] = str(
                        XmlAugmentationParser.__getDistValue(
                            float(value[0]), float(value[1])
                        )
                    )
                elif description == "Draw Fog":
                    value = str(random.choice(conf["draw_fog"])).lower()
                    param.attrib["Value"] = value
                elif description == "Output filename":
                    param.attrib["Value"] = img[:-4]
                elif description == "Pixel width":
                    param.attrib["Value"] = str(size[0])
                elif description == "Pixel height":
                    param.attrib["Value"] = str(size[1])
                elif description == "Output directory":
                    param.attrib["Value"] = kitti.aug_path

        tree.write(kitti.kitti_dir + "/augmentation/augment.xml")

    @staticmethod
    def __parseFog(conf, img, kitti):
        tree = ET.parse(kitti.kitti_dir + "/augmentation/fog.xml")
        root = tree.getroot()
        size = XmlAugmentationParser.__getImgSize(kitti.kitti_dir, img)

        for params in root.iter("ParameterList"):
            for param in params:
                description = param.attrib["Description"]
                if description == "Fog Density [1/um^3]":
                    value = conf["fog_density"]
                    param.attrib["Value"] = str(
                        XmlAugmentationParser.__getDistValue(
                            float(value[0]), float(value[1])
                        )
                    )
                elif description == "Fog Sphere Diameter [um]":
                    value = conf["fog_sphere"]
                    param.attrib["Value"] = str(
                        XmlAugmentationParser.__getDistValue(
                            float(value[0]), float(value[1])
                        )
                    )
                elif description == "Output filename":
                    param.attrib["Value"] = img[:-4]
                elif description == "Pixel width":
                    param.attrib["Value"] = str(size[0])
                elif description == "Pixel height":
                    param.attrib["Value"] = str(size[1])
                elif description == "Output directory":
                    param.attrib["Value"] = kitti.aug_path

        tree.write(kitti.kitti_dir + "/augmentation/augment.xml")

    @staticmethod
    def __imagecorruptions(conf, img, kitti):
        pil_img = Image.open(kitti.kitti_dir + "/training/image_2/" + img).convert(
            "RGB"
        )
        pil_img = corrupt(
            np.array(pil_img),
            corruption_name=conf["corruption"],
            severity=int(
                XmlAugmentationParser.__getDistValue(
                    int(conf["severity"][0]), int(conf["severity"][1])
                )
            ),
        )
        Image.fromarray(pil_img).save(kitti.aug_path + img)


class AugmentationThread:
    def __init__(self):
        self.stop = True
        self.running = False

    def call_augment(self, conf, img, kitti, out):
        if XmlAugmentationParser.parse(conf, img, kitti):
            subprocess.call(
                kitti.kitti_dir + "/augmentation/perform_augmentation.sh",
                stdout=subprocess.DEVNULL,
            )

        if out is True:
            print("Image augmented")

    def augment_img(self, kitti, img, conf, reaugment, out):
        if reaugment is True:
            txt = open(kitti.kitti_dir + "/augmentation/to_augment.txt", "w")
            txt.write(img[:-4] + "\n")
            txt.close()
            self.call_augment(conf, img, kitti, out)
        kitti.aug_files.append(img[:-4])

    def augment(self, conf, kitti, pct, aug_new, out):
        self.running = True
        self.stop = False
        reaugment = conf["reaugment_per_epoch_pct"]
        num_augment = len(kitti.img_files) * (pct / 100)

        for img in kitti.aug_files:
            # Remove reaugment_per_epoch_pct images from augmentation list
            if self.stop:
                break

            random.seed()
            rand = random.randrange(0, 100)

            if rand < reaugment:
                kitti.aug_files.remove(img[:-4])

        for img in kitti.img_files:
            # Add images to augmentation list to reach augmented_pct and restart augmentation if necessary
            if self.stop:
                break
            random.seed()
            rand = random.randrange(0, 100)

            if img[:-4] in kitti.aug_files and not os.path.isfile(kitti.aug_path + img):
                self.augment_img(kitti, img, conf, True, out)
            elif (
                rand < pct
                and len(kitti.aug_files) <= num_augment
                and img[:-4] not in kitti.aug_files
            ):
                self.augment_img(kitti, img, conf, aug_new, out)
        self.stop = True
        self.running = False

    def clear(self):
        self.stop = True

        while self.running:
            continue


class Augmentation:
    def __init__(self, augmentation: list()):
        self.aug_thread = AugmentationThread()
        self.conf = dict((key, a[key]) for a in augmentation for key in a)
        self.pct = self.conf["augmented_pct"] if "augmented_pct" in self.conf else 0
        self.bordersearch_epochs = self.conf["bordersearch_epoch_duration"]
        self.waterlevel = self.conf["bordersearch_waterlevel"]
        self.setEvalAttribs()

    def augment(self, kitti: Kitti):
        kitti.aug_files = list()

        if self.pct != 0 and self.val_pct != 0:
            self.aug_thread.clear()
            th = threading.Thread(
                target=self.aug_thread.augment,
                args=(
                    self.conf,
                    kitti,
                    self.pct if kitti.set_type == DatasetType.TRAIN else self.val_pct,
                    self.reaugment,
                    self.out,
                ),
                daemon=True,
            )
            th.start()

            if self.wait is True:
                print("######### WAIT FOR AUGMENTATION #########")
                th.join()

    def fillParams(self):
        ignore = self.conf["bordersearch_ignore_params"]
        params = list()
        i = 0

        if "rain_drops" in self.conf["augmentations"]:
            rain = dict((key, a[key]) for a in self.conf["rain_drops"] for key in a)
            for elem in list(rain):
                if elem not in ignore:
                    params.append(
                        Parameter(
                            ParameterRange(rain[elem][0], rain[elem][1]),
                            elem,
                            "rain_drops",
                            i,
                        )
                    )
                    i += 1
        if "fog" in self.conf["augmentations"]:
            fog = dict((key, a[key]) for a in self.conf["fog"] for key in a)
            for elem in list(fog):
                if elem not in ignore:
                    params.append(
                        Parameter(
                            ParameterRange(fog[elem][0], fog[elem][1]), elem, "fog", i
                        )
                    )
                    i += 1
        if "snow" in self.conf["augmentations"]:
            snow = dict((key, a[key]) for a in self.conf["snow"] for key in a)
            for elem in list(snow):
                if elem not in ignore:
                    params.append(
                        Parameter(
                            ParameterRange(snow[elem][0], snow[elem][1]),
                            elem,
                            "snow",
                            i,
                        )
                    )
                    i += 1
        if "imagecorruptions" in self.conf["augmentations"]:
            imagecorruptions = dict(
                (key, a[key]) for a in self.conf["imagecorruptions"] for key in a
            )
            for elem in list(imagecorruptions):
                if elem not in ignore:
                    params.append(
                        Parameter(
                            ParameterRange(
                                imagecorruptions[elem][0], imagecorruptions[elem][1]
                            ),
                            elem,
                            "imagecorruptions",
                            i,
                        )
                    )
                    i += 1

        return params

    def changeParams(self, params, conf):
        for param, value in zip(params, conf[0][0]):
            for c in self.conf[param.catuuid]:
                if c[param.uuid] is not None:
                    c[param.uuid][0] = value
                    break

    def setEvalAttribs(self, val_pct=50, wait=False, reaugment=True, out=False):
        self.val_pct = val_pct
        self.wait = wait
        self.reaugment = reaugment
        self.out = out
