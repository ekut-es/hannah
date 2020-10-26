import torch.utils.data as data
import torch
import os
import platform
import shutil
import random

from .dataset import ctc_collate_fn
from .utils import _locate
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.metrics.functional import accuracy, f1_score, recall
from .config_utils import (
    get_lr_scheduler,
    get_loss_function,
    get_optimizer,
    get_model,
    save_model,
)

from torchvision.datasets.utils import (
    download_and_extract_archive,
    extract_archive,
    list_files,
)


class SpeechClassifierModule(LightningModule):
    def __init__(self, config=None, log_dir="", msglogger=None):
        super().__init__()

        self.hparams = config

        # get all the necessary data stuff
        _locate(config["dataset_cls"]).download(config)
        self.download_noise(config)
        self.split_data(config)
        self.downsample(config)

        # trainset needed to set values in hparams
        self.train_set, self.dev_set, self.test_set = _locate(
            config["dataset_cls"]
        ).splits(config)
        self.hparams["width"] = self.train_set.width
        self.hparams["height"] = self.train_set.height
        self.model = get_model(self.hparams)

        # loss function
        self.criterion = get_loss_function(self.model, self.hparams)

        # logging
        self.log_dir = log_dir
        self.msglogger = msglogger

        self.msglogger.info("speech classifier initialized")

        # summarize model architecture
        dummy_width, dummy_height = self.train_set.width, self.train_set.height
        dummy_input = torch.zeros(1, dummy_height, dummy_width)
        self.example_input_array = dummy_input
        self.bn_frozen = False

    def split_data(self, config):
        data_split = config["data_split"]
        splits = ["vad", "vad_speech", "vad_balanced", "getrennt"]

        if data_split in splits:
            print("split data begins")
            data_folder = config["data_folder"]

            # remove old folders
            for name in ["train", "dev", "test"]:
                oldpath = os.path.join(data_folder, name)
                if os.path.isdir(oldpath):
                    shutil.rmtree(oldpath)

            # directories with original data
            noise_dir = os.path.join(data_folder, "noise_files")
            speech_dir = os.path.join(data_folder, "speech_files")

            if data_split == "vad_speech":
                speech_dir = os.path.join(data_folder, "speech_commands_v0.02")

            destination_dict = dict()

            # list all noise  and speech files
            speech_files_P = []
            speech_files = []
            noise_files = []
            for path, subdirs, files in os.walk(noise_dir):
                for name in files:
                    if (
                        name.endswith("wav") or name.endswith("mp3")
                    ) and not name.startswith("."):
                        noise_files.append(os.path.join(path, name))

            if (data_split == "vad") or (data_split == "vad_balanced"):
                for path, subdirs, files in os.walk(speech_dir):
                    for name in files:
                        if (
                            name.endswith("wav") or name.endswith("mp3")
                        ) and not name.startswith("."):
                            speech_files.append(os.path.join(path, name))
            elif data_split == "vad_speech":
                for path, subdirs, files in os.walk(speech_dir):
                    if "noise" not in subdirs:
                        for name in files:
                            if (
                                name.endswith("wav") or name.endswith("mp3")
                            ) and not name.startswith("."):
                                speech_files.append(os.path.join(path, name))
            elif data_split == "getrennt":
                speech_files_N = []
                for path, subdirs, files in os.walk(speech_dir):
                    for name in files:
                        if (
                            name.endswith("wav") or name.endswith("mp3")
                        ) and not name.startswith("."):
                            if "NC" in name:
                                speech_files_P.append(os.path.join(path, name))
                            else:
                                speech_files_N.append(os.path.join(path, name))

            # randomly shuffle the noise and speech files and split them in train,
            # validation and test set
            random.shuffle(noise_files)
            random.shuffle(speech_files)

            nb_noise_files = len(noise_files)
            nb_train_noise = int(0.6 * nb_noise_files)
            nb_dev_noise = int(0.2 * nb_noise_files)

            if "vad" == data_split:
                nb_speech_files = len(speech_files)
                nb_train_speech = int(0.6 * nb_speech_files)
                nb_dev_speech = int(0.2 * nb_speech_files)

                train_noise = noise_files[:nb_train_noise]
                dev_noise = noise_files[nb_train_noise : nb_train_noise + nb_dev_noise]
                test_noise = noise_files[nb_train_noise + nb_dev_noise :]

                train_speech = speech_files[:nb_train_speech]
                dev_speech = speech_files[
                    nb_train_speech : nb_train_speech + nb_dev_speech
                ]
                test_speech = speech_files[nb_train_speech + nb_dev_speech :]

                destination_dict = {
                    "train/noise": train_noise,
                    "train/speech": train_speech,
                    "dev/noise": dev_noise,
                    "dev/speech": dev_speech,
                    "test/noise": test_noise,
                    "test/speech": test_speech,
                }

            elif (
                ("vad_balanced" == data_split)
                or ("vad_speech" == data_split)
                or ("getrennt" == data_split)
            ):
                train_noise = noise_files[:nb_train_noise]
                dev_noise = noise_files[nb_train_noise : nb_train_noise + nb_dev_noise]
                test_noise = noise_files[nb_train_noise + nb_dev_noise :]

                if ("vad_balanced" == data_split) or ("vad_speech" == data_split):
                    train_speech = speech_files[:nb_train_noise]
                    dev_speech = speech_files[
                        nb_train_noise : nb_train_noise + nb_dev_noise
                    ]
                    test_speech = speech_files[
                        nb_train_noise + nb_dev_noise : nb_noise_files
                    ]
                elif "getrennt" == data_split:
                    random.shuffle(speech_files_P)
                    train_speech = speech_files_N[:nb_train_noise]
                    dev_speech = speech_files_P[
                        nb_train_noise : nb_train_noise + nb_dev_noise
                    ]
                    test_speech = speech_files_P[
                        nb_train_noise + nb_dev_noise : nb_noise_files
                    ]

                train_bg_noise = train_noise[:100]
                dev_bg_noise = dev_noise[:100]
                test_bg_noise = test_noise[:100]

                destination_dict = {
                    "train/noise": train_noise,
                    "train/speech": train_speech,
                    "dev/noise": dev_noise,
                    "dev/speech": dev_speech,
                    "test/noise": test_noise,
                    "test/speech": test_speech,
                    "train/background_noise": train_bg_noise,
                    "dev/background_noise": dev_bg_noise,
                    "test/background_noise": test_bg_noise,
                }

            for key, value in destination_dict.items():
                data_dir = os.path.join(data_folder, key)
                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)
                for f in value:
                    shutil.copy2(f, data_dir)

            if config["clear_split"] == "clear":
                # remove old folders
                for name in ["noise_files", "speech_files", "speech_commands_v0.02"]:
                    oldpath = os.path.join(data_folder, name)
                    if os.path.isdir(oldpath):
                        shutil.rmtree(oldpath)

    def downsample(self, config):
        samplerate = config["downsample"]
        if samplerate > 0:
            print("downsample data begins")
            downsample_folder = ["train", "dev", "test"]
            for folder in downsample_folder:
                folderpath = os.path.join(config["data_folder"], folder)
                if os.path.isdir(folderpath):
                    for path, subdirs, files in os.walk(folderpath):
                        for name in files:
                            if name.endswith("wav") and not name.startswith("."):
                                os.system(
                                    "ffmpeg -y -i "
                                    + os.path.join(path, name)
                                    + " -ar "
                                    + str(samplerate)
                                    + " -loglevel quiet "
                                    + os.path.join(path, "new" + name)
                                )
                                os.system("rm " + os.path.join(path, name))
                                os.system(
                                    "mv "
                                    + os.path.join(path, "new" + name)
                                    + " "
                                    + os.path.join(path, name)
                                )
                            elif name.endswith("mp3") and not name.startswith("."):
                                os.system(
                                    "ffmpeg -y -i "
                                    + os.path.join(path, name)
                                    + " -ar "
                                    + str(samplerate)
                                    + " -ac 1 -loglevel quiet "
                                    + os.path.join(path, name.replace(".mp3", ".wav"))
                                )
                                os.system("rm " + os.path.join(path, name))

    def download_noise(self, config):
        data_folder = config["data_folder"]
        clear_download = config["clear_download"]
        if not os.path.isdir(data_folder):
            os.makedirs(data_folder)
        noise_folder = os.path.join(data_folder, "noise_files")

        if not os.path.isdir(noise_folder):
            os.makedirs(noise_folder)
            noisedatasets = config["noise_dataset"].split("/")

            # Test if the the code is run on lucille or not
            if platform.node() == "lucille":
                # datasets are in /storage/local/dataset/...... prestored
                noisekeys = ["FSD/FSDKaggle", "FSD/FSDnoisy", "TUT", "FSD/FSD50K"]
                for key in noisekeys:
                    if key in noisedatasets:
                        source = os.path.join("/storage/local/dataset/", key)
                        mvtarget = os.path.join(noisedatasets, key)
                        os.makedirs(mvtarget)
                        for element in list_files(source, ".zip"):
                            file_to_extract = os.path.join(source, element)
                            extract_archive(file_to_extract, mvtarget, False)
            else:
                if "TUT" in noisedatasets:
                    for i in range(1, 10):
                        download_and_extract_archive(
                            "https://zenodo.org/record/400515/files/TUT-acoustic-scenes-2017-development.audio."
                            + str(i)
                            + ".zip",
                            noise_folder,
                            noise_folder,
                            remove_finished=clear_download,
                        )

                FSDParts = ["audio_test", "audio_train", "meta"]
                datasetname = ["FSDKaggle", "FSDnoisy"]
                FSDLinks = [
                    "https://zenodo.org/record/2552860/files/FSDKaggle2018.",
                    "https://zenodo.org/record/2529934/files/FSDnoisy18k.",
                ]
                for name, url in zip(datasetname, FSDLinks):
                    if name in noisedatasets:
                        targetfolder = os.path.join(noise_folder, name)
                        os.makedirs(targetfolder)
                        for part in FSDParts:
                            download_and_extract_archive(
                                url + part + ".zip",
                                targetfolder,
                                targetfolder,
                                remove_finished=clear_download,
                            )

    def configure_optimizers(self):
        optimizer = get_optimizer(self.hparams, self)
        scheduler = get_lr_scheduler(self.hparams, optimizer)

        return [optimizer], [scheduler]

    def get_batch_metrics(self, output, y, loss, prefix):

        # in case of multiple outputs
        if isinstance(output, list):
            # log for each output
            for idx, out in enumerate(output):
                # accuracy
                # self.log(f"{prefix}_acc_step/exit_{idx}", self.accuracy[idx](out, y))
                self.log(
                    f"{prefix}_accuracy/exit_{idx}",
                    accuracy(out, y),
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                )
                self.log(
                    f"{prefix}_recall/exit_{idx}",
                    recall(out, y),
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                )
                self.log(
                    f"{prefix}_f1/exit_{idx}",
                    f1_score(out, y),
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                )
            # TODO: f1 recall

        else:
            self.log(
                f"{prefix}_f1",
                f1_score(output, y),
                on_step=True,
                on_epoch=True,
                logger=True,
            )
            self.log(
                f"{prefix}_accuracy",
                accuracy(output, y),
                on_step=True,
                on_epoch=True,
                logger=True,
            )
            self.log(
                f"{prefix}_recall",
                recall(output, y),
                on_step=True,
                on_epoch=True,
                logger=True,
            )

        # only one loss allowed
        # also in case of branched networks
        self.log(f"{prefix}_loss", loss, on_step=True, on_epoch=True, logger=True)

    # TRAINING CODE
    def training_step(self, batch, batch_idx):
        x, x_len, y, y_len = batch
        output = self(x)
        y = y.view(-1)
        loss = self.criterion(output, y)

        # --- after loss
        for callback in self.trainer.callbacks:
            if hasattr(callback, "on_before_backward"):
                callback.on_before_backward(self.trainer, self, loss)
        # --- before backward

        # METRICS
        self.get_batch_metrics(output, y, loss, "train")

        return loss

    def train_dataloader(self):

        train_batch_size = self.hparams["batch_size"]
        train_loader = data.DataLoader(
            self.train_set,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=self.hparams["num_workers"],
            collate_fn=ctc_collate_fn,
        )

        self.batches_per_epoch = len(train_loader)

        return train_loader

    # VALIDATION CODE

    def validation_step(self, batch, batch_idx):

        # dataloader provides these four entries per batch
        x, x_length, y, y_length = batch

        # INFERENCE
        output = self.model(x)
        y = y.view(-1)
        loss = self.criterion(output, y)

        # METRICS
        self.get_batch_metrics(output, y, loss, "val")
        return loss

    def val_dataloader(self):

        dev_loader = data.DataLoader(
            self.dev_set,
            batch_size=min(len(self.dev_set), 16),
            shuffle=False,
            num_workers=self.hparams["num_workers"],
            collate_fn=ctc_collate_fn,
        )

        return dev_loader

    # TEST CODE
    def test_step(self, batch, batch_idx):

        # dataloader provides these four entries per batch
        x, x_length, y, y_length = batch

        output = self.model(x)
        y = y.view(-1)
        loss = self.criterion(output, y)

        # METRICS
        self.get_batch_metrics(output, y, loss, "test")

        return loss

    def test_dataloader(self):

        test_loader = data.DataLoader(
            self.test_set,
            batch_size=1,
            shuffle=False,
            num_workers=self.hparams["num_workers"],
            collate_fn=ctc_collate_fn,
        )

        return test_loader

    # FORWARD (overwrite to train instance of this class directly)
    def forward(self, x):
        return self.model(x)

    # CALLBACKS
    def on_train_end(self):
        # TODO currently custom save, in future proper configure lighting for saving ckpt
        save_model(
            self.log_dir,
            self.model,
            self.test_set,
            config=self.hparams,
            msglogger=self.msglogger,
        )
