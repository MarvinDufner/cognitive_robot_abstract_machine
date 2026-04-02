from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse
from zipfile import ZipFile
import json
import requests

from semantic_digital_twin.adapters.sage_10k_dataset.schema import Sage10kScene


@dataclass
class Sage10kDatasetLoader:
    """
    Loader for scenes from the Sage10k dataset.
    """

    directory: Path = field(default_factory=lambda: Path.home() / "sage-10k-scenes")
    """
    The directory where the scene should be downloaded to.
    """

    def _download_scene(self, scene_url: str) -> Path:
        """
        Download the scene from the Sage10k dataset and return it as a Path.
        :return: The path to the scene downloaded.
        """
        self.directory.mkdir(parents=True, exist_ok=True)

        filename = Path(urlparse(scene_url).path).name
        target_path = self.directory / filename

        # check if the target file already exists
        if target_path.exists():
            return target_path

        with requests.get(scene_url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with target_path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        return target_path

    def _unzip_scene(self, scene_path: Path) -> Path:
        """
        Extracts the contents of a zip file to a directory. If the directory already exists,
        the function will return the existing directory path without performing any operations.

        :param scene_path: Path to the zip file that needs to be extracted.
        :return: The path to the directory where the contents of the zip file have been extracted.
        """
        extract_dir = self.directory / scene_path.stem

        if extract_dir.exists():
            return extract_dir

        extract_dir.mkdir(parents=True, exist_ok=True)

        with ZipFile(scene_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        return extract_dir

    def _parse_json(self, extracted_dir: Path) -> Sage10kScene:
        """
        Parses the extracted directory to locate and load a specific JSON file, ensuring there is
        exactly one valid file matching the naming pattern. Load the JSON into a Sage10kScene object.

        :param extracted_dir: The directory containing the extracted files to be parsed.
        :return: A Sage10kScene object created from the parsed JSON content. The object's
            `directory_path` attribute is also updated to the given `extracted_dir`.
        """
        json_files = list(extracted_dir.glob("layout_*.json"))
        if not json_files:
            raise ValueError(f"JSON file not found in {extracted_dir}")
        elif len(json_files) > 1:
            raise ValueError(f"Multiple JSON files found in {extracted_dir}")
        json_file = json_files[0]

        raw_json = json_file.read_text()
        json_dict = json.loads(raw_json)
        result = Sage10kScene._from_json(json_dict)
        result.directory = extracted_dir

        return result

    def create_scene(self, scene_url: str) -> Sage10kScene:
        """
        Create a scene from the given URL by downloading it and loading it into the memory.

        :param scene_url: The URL of the scene to be loaded.
        :return: The Sage10kScene object.
        """
        target_path = self._download_scene(scene_url)
        unzipped = self._unzip_scene(target_path)
        scene = self._parse_json(unzipped)
        return scene

    @classmethod
    def available_scenes(
        cls, repository: str = "nvidia/SAGE-10k", folder_path: str = "scenes"
    ) -> list[str]:
        """
        Use this to select random scenes from the dataset.
        Requires the extra requirement huggingface_hu.

        :param repository: The repo id of the dataset.
        :param folder_path: The path to the folder containing the scenes in the repository.
        :return: A list of all possible URLs to the scenes in the dataset.
        """
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()

        # Hugging Face filesystem paths follow the format: datasets/repo_id/path
        full_path = f"datasets/{repository}/{folder_path}"

        # List all files (glob '**/*' to get nested files if needed)
        files = fs.glob(f"{full_path}/**/*")

        # Filter out directories and convert to "resolve" URLs
        base_url = f"https://huggingface.co/datasets/{repository}/resolve/main"

        urls = []
        for file in files:
            # fs.glob returns paths like 'datasets/nvidia/SAGE-10k/scenes/file.ext'
            # We need to strip the 'datasets/repo_id/' prefix
            relative_path = file.replace(f"datasets/{repository}/", "")
            urls.append(f"{base_url}/{relative_path}")

        return urls
