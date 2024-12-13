"""
This file houses the classes that load the footprint data from compressed, binary, and CSV files.
"""
import json
import logging
import mmap
import os
from contextlib import ExitStack
from typing import Dict, List, Union
from zlib import decompress

import numpy as np
import pandas as pd
import numba as nb

from oasis_data_manager.df_reader.config import clean_config, InputReaderConfig, get_df_reader
from oasis_data_manager.df_reader.reader import OasisReader
from oasis_data_manager.filestore.backends.base import BaseStorage
from .common import (FootprintHeader, EventIndexBin, EventIndexBinZ, Event, EventCSV,
                     footprint_filename, footprint_index_filename, zfootprint_filename, zfootprint_index_filename,
                     csvfootprint_filename, parquetfootprint_filename, parquetfootprint_meta_filename,
                     fp_format_priorities)

from .common import (ZCBinEvent, zcbin_filename, zcbin_index)
from oasislmf.pytools.gulmc.items import read_items
logger = logging.getLogger(__name__)

uncompressedMask = 1 << 1
intensityMask = 1


CURRENT_DIRECTORY = str(os.getcwd())


class OasisFootPrintError(Exception):
    """
    Raises exceptions when loading footprints.
    """

    def __init__(self, message: str) -> None:
        """
        The constructor of the OasisFootPrintError class.

        Args:
            message: (str) the message to be raised
        """
        super().__init__(message)


class Footprint:
    """
    This class is the base class for the footprint loaders.

    Attributes:
        storage (BaseStorage): the storage object used to lookup files
        stack (ExitStack): the context manager that combines other context managers and cleanup functions
    """

    def __init__(self, storage: BaseStorage, df_engine="oasis_data_manager.df_reader.reader.OasisPandasReader") -> None:
        """
        The constructor for the Footprint class.

        Args:
            storage (BaseStorage): the storage object used to lookup files
        """
        self.storage = storage
        self.stack = ExitStack()
        self.df_engine = df_engine

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stack.__exit__(exc_type, exc_value, exc_traceback)

    @staticmethod
    def get_footprint_fmt_priorities():
        """
        Get list of footprint file format classes in order of priority.

        Returns: (list) footprint file format classes
        """
        format_to_class = {
            'parquet': FootprintParquet, 'csv': FootprintCsv,
            'binZ': FootprintBinZ, 'bin': FootprintBin,
        }
        priorities = [format_to_class[fmt] for fmt in fp_format_priorities if fmt in format_to_class]

        return priorities

    @classmethod
    def load(
        cls,
        storage: BaseStorage,
        ignore_file_type=set(),
        df_engine="oasis_data_manager.df_reader.reader.OasisPandasReader",
    ):
        """
        Loads the loading classes defined in this file checking to see if the files are in the static path
        whilst doing so. The loading goes through the hierarchy with the following order:

        -> compressed binary file
        -> binary file
        -> CSV file

        If the compressed binary file is present, this will be loaded. If it is not, then the binary file will be loaded
        and so on.

        Args:
            storage (BaseStorage): the storage object used to lookup files
            ignore_file_type (Set[str]): type of file to be skipped in the hierarchy. This can be a choice of:

            parquet
            json
            z
            bin
            idx

        Returns: (Union[FootprintBinZ, FootprintBin, FootprintCsv]) the loaded class
        """
        for footprint_class in cls.get_footprint_fmt_priorities():
            for filename in footprint_class.footprint_filenames:
                if (not storage.exists(filename)
                        or filename.rsplit('.', 1)[-1] in ignore_file_type):
                    valid = False
                    break
            else:
                valid = True
            if valid:
                for filename in footprint_class.footprint_filenames:
                    logger.debug(f"loading {filename}")
                return footprint_class(storage, df_engine=df_engine)
        else:
            if storage.isfile("footprint.parquet"):
                raise OasisFootPrintError(
                    message="footprint.parquet needs to be partitioned in order to work, please see: "
                            "oasislmf.pytools.data_layer.conversions.footprint => convert_bin_to_parquet"
                )
            raise OasisFootPrintError(message="no valid footprint found")

    def get_event(self, event_id):
        raise NotImplementedError()

    def get_df_reader(self, filepath, **kwargs) -> OasisReader:
        # load the base df engine config and add the connection parameters
        df_reader_config = clean_config(InputReaderConfig(
            filepath=filepath,
            engine=self.df_engine
        ))
        df_reader_config["engine"]["options"]["storage"] = self.storage

        return get_df_reader(df_reader_config, **kwargs)

    @staticmethod
    def prepare_df_data(data_frame: pd.DataFrame) -> np.array:
        """
        Reads footprint data from a parquet file.

        Returns: (np.array) footprint data loaded from the parquet file
        """
        areaperil_id = data_frame["areaperil_id"].to_numpy()
        intensity_bin_id = data_frame["intensity_bin_id"].to_numpy()
        probability = data_frame["probability"].to_numpy()

        buffer = np.empty(len(areaperil_id), dtype=Event)
        outcome = stitch_data(areaperil_id, intensity_bin_id, probability, buffer)
        return np.array(outcome, dtype=Event)


class FootprintCsv(Footprint):
    """
    This class is responsible for loading footprint data from CSV.

    Attributes (when in context):
        footprint (np.array[EventCSV]): event data loaded from the CSV file
        num_intensity_bins (int): number of intensity bins in the data
        has_intensity_uncertainty (bool): if the data has uncertainty
        footprint_index (dict): map of footprint IDs with the index in the data
    """
    footprint_filenames = [csvfootprint_filename]

    def __enter__(self):
        self.reader = self.get_df_reader("footprint.csv", dtype=EventCSV)

        self.num_intensity_bins = self.reader.query(lambda df: df['intensity_bin_id'].max())

        self.has_intensity_uncertainty = self.reader.query(
            lambda df: df.groupby(['event_id', 'areaperil_id']).size().max() > 1
        )

        def _fn(df):
            footprint_index_df = df.groupby('event_id', as_index=False).size()
            footprint_index_df['offset'] = footprint_index_df['size'].cumsum() - footprint_index_df['size']
            footprint_index_df.set_index('event_id', inplace=True)
            return footprint_index_df
        self.footprint_index = self.reader.query(_fn).to_dict('index')

        return self

    def get_event(self, event_id):
        """
        Gets the event from self.footprint based off the event ID passed in.

        Args:
            event_id: (int) the ID belonging to the Event being extracted

        Returns: (np.array[EventCSV]) the event that was extracted
        """
        event_info = self.footprint_index.get(event_id)
        if event_info is None:
            return
        else:
            return self.prepare_df_data(self.reader.filter(lambda df: df[df["event_id"] == event_id]).as_pandas())


class FootprintBin(Footprint):
    """
    This class is responsible loading the event data from the footprint binary files.

    Attributes (when in context):
        footprint (mmap.mmap): loaded data from the binary file which has header and then Event data
        num_intensity_bins (int): number of intensity bins in the data
        has_intensity_uncertainty (bool): if the data has uncertainty
        footprint_index (dict): map of footprint IDs with the index in the data
    """
    footprint_filenames = [footprint_filename, footprint_index_filename]

    def __enter__(self):
        footprint_file = self.stack.enter_context(self.storage.with_fileno(footprint_filename))
        self.footprint = mmap.mmap(footprint_file.fileno(), length=0, access=mmap.ACCESS_READ)
        footprint_header = np.frombuffer(bytearray(self.footprint[:FootprintHeader.size]), dtype=FootprintHeader)
        self.num_intensity_bins = int(footprint_header['num_intensity_bins'])
        self.has_intensity_uncertainty = int(footprint_header['has_intensity_uncertainty'] & intensityMask)

        f = self.stack.enter_context(self.storage.with_fileno(footprint_index_filename))
        footprint_mmap = np.memmap(f, dtype=EventIndexBin, mode='r')

        self.footprint_index = pd.DataFrame(
            footprint_mmap,
            columns=footprint_mmap.dtype.names
        ).set_index('event_id').to_dict('index')
        return self

    def get_event(self, event_id):
        """
        Gets the event from self.footprint based off the event ID passed in.

        Args:
            event_id: (int) the ID belonging to the Event being extracted

        Returns: (np.array(Event)) the event that was extracted
        """
        event_info = self.footprint_index.get(event_id)
        if event_info is None:
            return
        else:
            return np.frombuffer(self.footprint[event_info['offset']: event_info['offset'] + event_info['size']], Event)


class FootprintBinZ(Footprint):
    """
    This class is responsible for loading event data from compressed event data.

    Attributes (when in context):
        zfootprint (mmap.mmap): loaded data from the compressed binary file which has header and then Event data
        num_intensity_bins (int): number of intensity bins in the data
        has_intensity_uncertainty (bool): if the data has uncertainty
        uncompressed_size (int): the size in which the data is when it is decompressed
        index_dtype (Union[EventIndexBinZ, EventIndexBin]) the data type
        footprint_index (dict): map of footprint IDs with the index in the data
    """
    footprint_filenames = [zfootprint_filename, zfootprint_index_filename]

    def __enter__(self):
        zfootprint_file = self.stack.enter_context(self.storage.with_fileno(zfootprint_filename))
        self.zfootprint = mmap.mmap(zfootprint_file.fileno(), length=0, access=mmap.ACCESS_READ)

        footprint_header = np.frombuffer(bytearray(self.zfootprint[:FootprintHeader.size]), dtype=FootprintHeader)
        self.num_intensity_bins = int(footprint_header['num_intensity_bins'])
        self.has_intensity_uncertainty = int(footprint_header['has_intensity_uncertainty'] & intensityMask)
        self.uncompressed_size = int((footprint_header['has_intensity_uncertainty'] & uncompressedMask) >> 1)
        if self.uncompressed_size:
            self.index_dtype = EventIndexBinZ
        else:
            self.index_dtype = EventIndexBin

        f = self.stack.enter_context(self.storage.with_fileno(zfootprint_index_filename))
        zfootprint_mmap = np.memmap(f, dtype=self.index_dtype, mode='r')
        self.footprint_index = pd.DataFrame(zfootprint_mmap, columns=zfootprint_mmap.dtype.names).set_index('event_id').to_dict('index')
        return self

    def get_event(self, event_id):
        """
        Gets the event from self.zfootprint based off the event ID passed in.

        Args:
            event_id: (int) the ID belonging to the Event being extracted

        Returns: (np.array[Event]) the event that was extracted
        """
        event_info = self.footprint_index.get(event_id)
        if event_info is None:
            return
        else:
            zdata = self.zfootprint[event_info['offset']: event_info['offset'] + event_info['size']]
            data = decompress(zdata)
            return np.frombuffer(data, Event)


class FootprintParquet(Footprint):
    """
    This class is responsible for loading event data from parquet event data.

    Attributes (when in context):
        num_intensity_bins (int): number of intensity bins in the data
        has_intensity_uncertainty (bool): if the data has uncertainty
        footprint_index (dict): map of footprint IDs with the index in the data
    """
    footprint_filenames: List[str] = [parquetfootprint_filename, parquetfootprint_meta_filename]

    def __enter__(self):
        with self.storage.open(parquetfootprint_meta_filename, 'r') as outfile:
            meta_data: Dict[str, Union[int, bool]] = json.load(outfile)

        self.num_intensity_bins = int(meta_data['num_intensity_bins'])
        self.has_intensity_uncertainty = int(meta_data['has_intensity_uncertainty'] & intensityMask)

        return self

    def get_event(self, event_id: int):
        """
        Gets the event data from the partitioned parquet data file.

        Args:
            event_id: (int) the ID belonging to the Event being extracted

        Returns: (np.array[Event]) the event that was extracted
        """
        reader = self.get_df_reader("footprint.parquet", filters=[("event_id", "==", event_id)])

        numpy_data = self.prepare_df_data(data_frame=reader.as_pandas())
        return numpy_data


@nb.njit(nb.int64(nb.uint64), inline='always')
def get_level(areaperil_id):
    return np.ceil((np.log(areaperil_id)/np.log(4)))

@nb.njit(cache=True)
def refactor_fp(footprint, items):        
    # create the default QK
    null_qk = np.empty(1, dtype=ZCBinEvent)
    null_qk[0]['areaperil_id'] = 0
    null_qk[0]['intensity_bin_id'] = 1


    # index footprint
    contains_subperil = np.full(16, False, dtype=np.bool_)
    footprint_level_idx = np.zeros(21, dtype=np.int64)
    new_fp = np.empty(items.shape[0], dtype=Event)


    level_idx = 0
    init_apid = footprint[0]["areaperil_id"]
    prev_level = get_level(init_apid)
    for apid_idx in range(footprint.shape[0]):
        apid = footprint[apid_idx]["areaperil_id"]
        peril = np.int64(apid) & np.int64(15)
        contains_subperil[peril - 1] = True
        level = get_level(apid)
        if (level > prev_level):
            prev_level = level
            footprint_level_idx[level_idx] = apid_idx
            level_idx += 1  
        elif (level < prev_level):
            raise Exception("FP not sorted correctly")
        
    footprint_level_idx[level_idx] = footprint.shape[0]
    level_idx += 1
    
    # reindex the footprint
    footprint_i = 0
    for i in range (items.shape[0]):
        item = items[i]
        areaperil = item['areaperil_id']
        parent_qk = null_qk[0] 

        while (True):
            #binary search for the item in the footprint
            peril = np.int64(areaperil) & np.int64(15)

            if (contains_subperil[peril-1] == False):
                break
            
            ap_found = False
            level_idx = 0  
            lower = 0
            upper = footprint_level_idx[0] - 1
            mid = int((lower + upper)/2)


            while (True):
                fp_lvl = get_level(footprint[mid]["areaperil_id"])
                shift = 21 - fp_lvl # temporary patch 
                item_apid_shift = np.int64((np.int64(areaperil) >> np.int64((shift * 2 + 4))) << np.int64(4)) + np.int64(peril)
                while(True): # binary search 
                    mid_apid = footprint[mid]["areaperil_id"]

                    if (upper < lower):
                        break

                    if (mid_apid == item_apid_shift):
                        ap_found = True
                        break
                    elif (mid_apid < item_apid_shift):
                        lower = mid + 1
                    elif (mid_apid > item_apid_shift):
                        upper = mid - 1
                    
                    mid = int((lower + upper) / 2)

                if (ap_found == True):
                    break
                elif (level_idx < footprint_level_idx.shape[0] - 1): 
                    level_idx += 1
                    lower = footprint_level_idx[level_idx -1]
                    upper = footprint_level_idx[level_idx] - 1
                    mid = int((lower + upper) / 2)
                else:
                    break

            if (ap_found):
                parent_qk = footprint[mid]
            break
        

        # FIX 20241204: fix empty does not initliaze so "in" must be used ove the initilised values
        if item['areaperil_id'] not in new_fp['areaperil_id'][:footprint_i]:
            # replace the hit QK with the item qk so it appears correct to the compenents its passed to 
            new_fp[footprint_i]['areaperil_id'] = item['areaperil_id']
            new_fp[footprint_i]['intensity_bin_id'] = parent_qk['intensity_bin_id']
            new_fp[footprint_i]['probability'] = 1.0
            footprint_i += 1

    return new_fp[:footprint_i]

class CombusZCBIN(Footprint):
    footprint_filenames: List[str] = [zcbin_filename, zcbin_index]
    ALM = os.environ.get('MODEL', 'ALM') == 'ALM'

    def __enter__(self):
        zcbin_file = self.stack.enter_context(self.storage.with_fileno(zcbin_filename))
        self.fp = mmap.mmap(zcbin_file.fileno(), length=0, access=mmap.ACCESS_READ)
        self.filesize = self.fp.size()
        
        footprint_header = np.frombuffer(bytearray(self.fp[:FootprintHeader.size]), dtype=FootprintHeader)
        self.num_intensity_bins = int(footprint_header['num_intensity_bins'])
        self.has_intensity_uncertainty = int(footprint_header['has_intensity_uncertainty'] & intensityMask)

        # read in the index file and convert to dataframe
        zcbin_idx = self.stack.enter_context(self.storage.with_fileno(zcbin_index))
        footprint_mmap = np.memmap(zcbin_idx, dtype=EventIndexBin, mode='r')

        self.footprint_index = pd.DataFrame(
            footprint_mmap,
            columns=footprint_mmap.dtype.names
        )
        # self.footprint_index.to_csv("fp_index.csv")
        self.footprint_index = self.footprint_index.set_index('event_id').to_dict('index')
        self.footprint_key_idx  = list(self.footprint_index)

        #### new methodology 
        # insteaf of reading in the event and then using a qk binary search, we will read in the event and then
        # parse the footprint using the items file so the footprint and items fil are matching
        run_dir = '.' # this is the default it seems for the oasislmf runs since the call to footprint comes from the run dir 
        input_path = os.path.join(run_dir, 'input')
        self.items = read_items(input_path)

        return self

    def unpack_event(self, event_id: int):
        event_info = self.footprint_index.get(event_id)
        if event_info is None:
            return
        else:
            if (self.footprint_key_idx.index(event_id) + 1) < len(self.footprint_key_idx):
                next_offset = self.footprint_index.get(self.footprint_key_idx[self.footprint_key_idx.index(event_id) + 1])
                next_offset = next_offset['offset']
            else:
                next_offset = self.filesize

            zdata = self.fp[event_info['offset']: next_offset]
            data = decompress(zdata)
            self.footprint = np.frombuffer(data, ZCBinEvent)

        
    def get_event(self, event_id):
        ### This is a slightly slower since we are stitching the data together first
        self.unpack_event(event_id)
        fp = refactor_fp(self.footprint, self.items)
        return fp
    

    
@nb.njit(cache=True)
def stitch_data(areaperil_id, intensity_bin_id, probability, buffer):
    """
    Creates a list of tuples from three np.arrays (all inputs must be the same length).
    Args:
        areaperil_id: (np.array) list of areaperil IDs
        intensity_bin_id: (np.array) list of probability bin IDs
        probability: (np.array) list of probabilities
        buffer: (np.array[Event]) list of zeros to be populated with the previous lists
    Returns:
    """
    for x in range(0, len(buffer)):
        buffer[x]['areaperil_id'] = areaperil_id[x]
        buffer[x]['intensity_bin_id'] = intensity_bin_id[x]
        buffer[x]['probability'] = probability[x]
    return buffer
