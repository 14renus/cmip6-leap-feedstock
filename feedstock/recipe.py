import glob
import logging
import os
import re
import shutil  # For cleanup
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, TypeVar, Union

import apache_beam as beam
import xarray as xr
from apache_beam.options.pipeline_options import PipelineOptions
from leap_data_management_utils.cmip_transforms import (
    Preprocessor,
    dynamic_chunking_func,
)
from pangeo_forge_recipes.patterns import pattern_from_file_sequence
from pangeo_forge_recipes.transforms import (
    ConsolidateDimensionCoordinates,
    # CheckpointFileTransfer,
    ConsolidateMetadata,
    OpenURLWithFSSpec,
    OpenWithXarray,
    StoreToZarr,
)

# Set up logging for Beam within the notebook
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
# Suppress xarray warning about unclosed files which can be common with many small files
xr.set_options(warn_for_unclosed_files=False)


def extract_parent_stem(f, n=8):
    """
    Extracts the parent directory names up to n levels up from the file path.
    Returns a string with the directory names joined by '/'.
    E.g., for a file path like '/path/to/my/data/file.nc', it returns
    '/to/my/data' if n=3.
    """
    stems = []
    for i in range(n):
        stem = Path(f).parent.name
        stems.append(stem)
        f = Path(f).parent

    return "/".join(stems[::-1])


def get_subdirectories_by_pattern(directory_pattern):
    """
    Gets directories matching a specific pattern.
    E.g., './my_data_root/project_*' to find 'project_A', 'project_B'.
    """
    matched_paths = glob.glob(directory_pattern)
    directories = [p for p in matched_paths if os.path.isdir(p) and ".zarr" not in p]
    return directories


def get_file_paths(directory, file_extension="*.nc"):
    """
    Generates file paths for xarray datasets in the given directory.
    Adjust file extension pattern as needed (e.g., '*.nc', '*.grib', '*.zarr').
    For this example, we'll look for NetCDF files.
    """
    search_pattern = os.path.join(directory, "**", file_extension)
    file_paths = glob.glob(search_pattern, recursive=True)
    if not file_paths:
        logging.warning(
            f"No files found matching '{search_pattern}' in directory '{directory}'. "
            "Please ensure your data files are present and match the pattern."
        )
    return file_paths


def open_with_xarray(filepath: str, load: bool = False, xarray_open_kwargs: Optional[dict] = None) -> xr.Dataset:
    """
    Reads an xarray Dataset from a given file path.
    Returns a tuple: (filepath, dataset).
    TODO: Should we use dask for lazy loading to prevent immediate memory issues.
    """
    # Use dask for lazy loading to handle large datasets more efficiently
    dataset = xr.open_dataset(filepath)  # , chunks='auto')
    if load:
        dataset.load()
    logging.debug(f"Successfully read dataset from {filepath}")
    return dataset


@dataclass
class OpenWithXarray(beam.PTransform):
    """Open indexed items with Xarray. Accepts either fsspec open-file-like objects
    or string URLs that can be passed directly to Xarray.

    :param file_type: Provide this if you know what type of file it is.
    :param load: Whether to eagerly load the data into memory ofter opening.
    :param copy_to_local: Whether to copy the file-like-object to a local path
       and pass the path to Xarray. Required for some file types (e.g. Grib).
       Can only be used with file-like-objects, not URLs.
    :param xarray_open_kwargs: Extra arguments to pass to Xarray's open function.
    """

    # file_type: FileType = FileType.unknown
    load: bool = False
    copy_to_local: bool = False
    xarray_open_kwargs: Optional[dict] = field(default_factory=dict)

    def expand(self, pcoll):
        return pcoll | "Open with Xarray" >> beam.MapTuple(
            lambda k, v: (
                k,
                open_with_xarray(
                    v,
                    # file_type=self.file_type,
                    load=self.load,
                    # copy_to_local=self.copy_to_local,
                    xarray_open_kwargs=self.xarray_open_kwargs,
                ),
            )
        )


recipes = {}
input_directory = "/usr/local/google/home/singhren/coding/data/cmip6/raw/CMIP6/CMIP/*/*/*/*/*/*"

for dir in get_subdirectories_by_pattern(input_directory):
    logger.info(dir)
    file_paths = sorted(get_file_paths(dir))
    output_zarr_file = re.sub(r"_\d{6}-\d{6}.nc$", ".zarr", Path(file_paths[0]).name)
    output_zarr_path = Path(extract_parent_stem(file_paths[0], n=8)) / output_zarr_file

    logger.info(f"File paths found: {file_paths}")
    pattern = pattern_from_file_sequence(file_paths, concat_dim="time")
    for k, v in pattern.items():
        logger.info(f"Pattern item: {k} -> {v}")

    recipes[dir] = (
        f"Creating {dir}" >> beam.Create(pattern.items())
        | OpenWithXarray(xarray_open_kwargs={"use_cftime": True})
        | Preprocessor()
        | StoreToZarr(
            # target_root=dir.replace("raw", "zarr"),
            store_name=str(output_zarr_path),
            combine_dims=pattern.combine_dim_keys,
            dynamic_chunking_fn=dynamic_chunking_func,
        )
        | ConsolidateDimensionCoordinates()  # Consolidate into one chunk.
        | ConsolidateMetadata()
    )
