"""pytest definitions to run the unittests."""

from __future__ import annotations

import builtins
import json
import shutil
from datetime import datetime
from pathlib import Path
from subprocess import PIPE, run
from tempfile import TemporaryDirectory
from typing import Generator, Optional, Union

import mock
import numpy as np
import pandas as pd
import pytest
import xarray as xr

PYSLK_DEFAULT_LIST_COLUMNS = [
    "permissions",
    "owner",
    "group",
    "filesize",
    "timestamp",
    "filename",
]


class SLKMock:
    """A mock that emulates what pyslk is doing."""

    def __init__(self, _cache: dict[int, builtins.list[str]] = {}) -> None:
        self._cache = _cache

    def slk_list(self, inp_path: str) -> str:
        """Mock the slk_list method."""
        res = (
            run(["ls", "-l", inp_path], stdout=PIPE, stderr=PIPE)
            .stdout.decode()
            .split("\n")
        )
        return "\n".join(res[1:] + [res[0]])

    def search(self, inp_f: list[str]) -> int | None:
        """Mock slk_search."""
        if not inp_f:
            return None
        hash_value = abs(hash(",".join(inp_f)))
        self._cache[hash_value] = inp_f
        return hash_value

    def gen_file_query(self, resources: list[str], **kwargs) -> list[str]:
        """Mock slk_gen_file_query."""
        return [f for f in resources if Path(f).exists()]

    def retrieve(
        self,
        resource: int,
        dest_dir: str,
        recursive: bool = False,
        group: Union[bool, None] = None,
        delayed: bool = False,
        preserve_path: bool = True,
        **kwargs,
    ) -> None:
        """Mock slk_retrieve."""
        for inp_file in map(Path, self._cache[resource]):
            if preserve_path:
                outfile = Path(dest_dir) / Path(str(inp_file).strip(inp_file.root))
                outfile.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(inp_file, outfile)
            else:
                shutil.copy(inp_file, Path(dest_dir) / inp_file.name)

    def ls(
        self,
        path_or_id: Union[
            str,
            int,
            Path,
            list[str],
            list[int],
            list[Path],
            set[str],
            set[int],
            set[Path],
        ],
        show_hidden: bool = False,
        numeric_ids: bool = False,
        recursive: bool = False,
        column_names: list = PYSLK_DEFAULT_LIST_COLUMNS,
        parse_dates: bool = True,
        parse_sizes: bool = True,
        full_path: bool = True,
    ) -> pd.DataFrame:
        """Mock slk_ls."""
        return pd.DataFrame(
            [
                [
                    "-rwxr-xr-x-",
                    "k204221",
                    "bm0146",
                    1268945,
                    datetime.now(),
                    "/arch/bm0146/k204221/iow/INDEX.txt",
                ]
            ],
            columns=column_names,
        )

    def group_files_by_tape(
        self,
        resource_path: Union[Path, str, list, None] = None,
        resource_ids: Union[str, list, int, None] = None,
        search_id: Union[str, int, None] = None,
        search_query: Union[str, None] = None,
        recursive: bool = False,
        max_tape_number_per_search: int = -1,
        run_search_query: bool = False,
        evaluate_regex_in_input: bool = False,
    ) -> list[dict]:
        """Mock slk_group_files_by_tape."""
        result = []
        if isinstance(resource_path, list):
            for path in resource_path:
                result.append(
                    {
                        "id": -1,
                        "location": "tape",
                        "barcode": "TEST_TAPE",
                        "status": "AVAILABLE",
                        "file_count": 1,
                        "files": [path],
                        "file_ids": [],
                    }
                )
        if isinstance(resource_path, (Path, str)):
            result.append(
                {
                    "id": -1,
                    "location": "tape",
                    "barcode": "TEST_TAPE",
                    "status": "AVAILABLE",
                    "file_count": 1,
                    "files": [path],
                    "file_ids": [],
                }
            )

        return result

    def get_tape_status(self, tape: int | str, details: bool = False) -> str | None:
        return "AVAILABLE"

    def recall_single(
        self,
        resources: (
            Path
            | str
            | int
            | list[Path]
            | list[str]
            | list[int]
            | set[Path]
            | set[str]
            | set[int]
        ),
        destination: Path | str | None = None,
        resource_ids: bool = False,
        search_id: bool = False,
        recursive: bool = False,
        preserve_path: bool = True,
    ) -> int:
        job_id: int = 12345
        return job_id

    def get_resource_tape(self, resource_path: str | Path) -> dict[int, str] | None:
        return {99999: "X9999999"}

    def get_resource_path(self, resource_id: str | int) -> Path | None:
        return Path("/test/precip.zarr")

    def retrieve_improved(
        self,
        resources: (
            Path
            | str
            | int
            | list[Path]
            | list[str]
            | list[int]
            | set[Path]
            | set[str]
            | set[int]
        ),
        destination: Path | str,
        dry_run: bool = False,
        force_overwrite: bool = False,
        ignore_existing: bool = False,
        resource_ids: bool = False,
        search_id: bool = False,
        recursive: bool = False,
        stop_on_failed_retrieval: bool = False,
        preserve_path: bool = True,
        verbose: bool = False,
    ) -> Optional[dict] | None:
        resource: str
        if isinstance(resources, (Path, str, int)):
            resource = str(resources)
        elif isinstance(resources, list):
            resource = str(resources[0])
        elif isinstance(resources, set):
            resource = str(resources.pop())
        output = json.loads(
            f"""
            {{
                "SKIPPED": {{"SKIPPED_TARGET_EXISTS": ["{resource}"]}},
                "FILES": {{"{resource}": "{str(destination)}"}}
            }}
            """
        )
        self._cache[abs(hash(resource))] = [resource]

        return output

    class PySlkException(BaseException):
        pass


def create_data(variable_name: str, size: int) -> xr.Dataset:
    """Create a xarray dataset."""
    coords: dict[str, np.ndarray] = {}
    coords["x"] = np.linspace(-10, -5, size)
    coords["y"] = np.linspace(120, 125, size)
    lat, lon = np.meshgrid(coords["y"], coords["x"])
    lon_vec = xr.DataArray(lon, name="Lg", coords=coords, dims=("y", "x"))
    lat_vec = xr.DataArray(lat, name="Lt", coords=coords, dims=("y", "x"))
    coords["time"] = np.array(
        [
            np.datetime64("2020-01-01T00:00"),
            np.datetime64("2020-01-01T12:00"),
            np.datetime64("2020-01-02T00:00"),
            np.datetime64("2020-01-02T12:00"),
        ]
    )
    dims = (4, size, size)
    data_array = np.empty(dims)
    for time in range(dims[0]):
        data_array[time] = np.zeros((size, size))
    dset = xr.DataArray(
        data_array,
        dims=("time", "y", "x"),
        coords=coords,
        name=variable_name,
    )
    data_array = np.zeros(dims)
    return xr.Dataset({variable_name: dset, "Lt": lon_vec, "Lg": lat_vec}).set_coords(
        list(coords.keys())
    )


@pytest.fixture(scope="session")
def patch_dir() -> Generator[Path, None, None]:
    with TemporaryDirectory() as temp_dir:
        with mock.patch("slkspec.core.pyslk", SLKMock()):
            yield Path(temp_dir)


@pytest.fixture(scope="session")
def save_dir() -> Generator[Path, None, None]:
    """Create a temporary directory."""
    with TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture(scope="session")
def data() -> Generator[xr.Dataset, None, None]:
    """Define a simple dataset with a blob in the middle."""
    dset = create_data("precip", 100)
    yield dset


@pytest.fixture(scope="session")
def netcdf_files(
    data: xr.Dataset,
) -> Generator[Path, None, None]:
    """Save data with a blob to file."""

    with TemporaryDirectory() as td:
        for time in (data.time[:2], data.time[2:]):
            time1 = pd.Timestamp(time.values[0]).strftime("%Y%m%d%H%M")
            time2 = pd.Timestamp(time.values[1]).strftime("%Y%m%d%H%M")
            out_file = (
                Path(td)
                / "the_project"
                / "test1"
                / "precip"
                / f"precip_{time1}-{time2}.nc"
            )
            out_file.parent.mkdir(exist_ok=True, parents=True)
            data.sel(time=time).to_netcdf(out_file)
        yield Path(td)


@pytest.fixture(scope="session")
def zarr_file(
    data: xr.Dataset,
) -> Generator[Path, None, None]:
    """Save data with a blob to file."""

    with TemporaryDirectory() as td:
        out_file = Path(td) / "the_project" / "test1" / "precip" / "precip.zarr"
        out_file.parent.mkdir(exist_ok=True, parents=True)
        data.to_zarr(out_file)
        yield Path(td)
