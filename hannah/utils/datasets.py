#
# Copyright (c) 2024 Hannah contributors.
#
# This file is part of hannah.
# See https://github.com/ekut-es/hannah for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import logging

from typing import Any
from torchvision.datasets.utils import (
    download_and_extract_archive,
    extract_archive,
    list_dir,
    list_files,
)

logger = logging.getLogger(__name__)


def list_all_files(
    path, file_suffix, file_prefix=False, remove_file_beginning=""
) -> Any:
    subfolder = list_dir(path, prefix=True)
    files_in_folder = list_files(path, file_suffix, prefix=file_prefix)
    for subfold in subfolder:
        subfolder.extend(list_dir(subfold, prefix=True))
        if len(remove_file_beginning):
            tmp = list_files(subfold, file_suffix, prefix=False)
            tmp = [
                element
                for element in tmp
                if not element.startswith(remove_file_beginning)
            ]
            for filename in tmp:
                files_in_folder.append(os.path.join(subfold, filename))
        else:
            files_in_folder.extend(list_files(subfold, file_suffix, prefix=file_prefix))

    return files_in_folder


def extract_from_download_cache(
    filename,
    url,
    cached_files,
    target_cache,
    target_folder,
    target_test_folder="",
    clear_download=False,
    no_exist_check=False,
) -> None:
    """extracts given file from cache or donwloads first from url

    Args:
        filename (str): name of the file to download or extract
        url (str): possible url to download the file
        cached_files (list(str)): cached files in download cache
        target_cache (str): path to the folder to cache file if download necessary
        target_folder (str): path where to extract file
        target_test_folder (str, optional): folder to check if data are already there
        clear_download (bool): clear download after usage
        no_exist_check (bool): disables the check if folder exists
    """
    if len(target_test_folder) == 0:
        target_test_folder = target_folder
    if filename not in cached_files and (
        not os.path.isdir(target_test_folder) or no_exist_check
    ):
        logger.info("download and extract: %s", str(filename))

        download_and_extract_archive(
            url,
            target_cache,
            target_folder,
            filename=filename,
            remove_finished=clear_download,
        )
    elif filename in cached_files and (
        not os.path.isdir(target_test_folder) or no_exist_check
    ):
        logger.info("extract from download_cache: %s", str(filename))

        extract_archive(
            os.path.join(target_cache, filename),
            target_folder,
            remove_finished=clear_download,
        )
