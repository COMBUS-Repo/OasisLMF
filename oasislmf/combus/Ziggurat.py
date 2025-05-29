import numpy as np
import numba as nb
import pandas as pd

import mmap
import os

from zlib_ng import zlib_ng
from numba.types import uint32, uint64, boolean


from oasislmf.pytools.getmodel.footprint import Footprint
from oasislmf.pytools.getmodel.footprint import FootprintBin
from oasislmf.pytools.getmodel.footprint import intensityMask
from oasislmf.pytools.getmodel.common import (FootprintHeader, EventIndexBin, Event, Item)

from .Quadkey import *

zcbin_filename = 'footprint.zcbin'
zcbin_index    = 'footprint.zcidx'

NUM_SUBPERILS = 16
NUM_LEVELS    = 18

##### Since we can overload this we dont need to (hopefully) need to update
# as much outside of the combus files and into the oasislmf code base
Footprint.get_footprint_fmt_priorities = staticmethod(lambda: [CombusZCBIN, FootprintBin])

def load(cls, storage_or_static_path, ignore_file_type=set(), *args, **kwargs):
    for footprint_class in cls.get_footprint_fmt_priorities():
        for filename in footprint_class.footprint_filenames:

            #? Assuming that we only need to support for v2 we dont need 
            #? this logic but has been left in comment so that if we do 
            #? we can back implement it to support v1.28.x 
            # Check if the file exists in the storage or static path
            # ensuring compatibility between versions
            storage_exists = False
            if isinstance(storage_or_static_path, str): # 1.28.x
                storage_exists = storage_or_static_path.endswith(filename)
            elif hasattr(storage_or_static_path, 'exists'): # ^2.3.x
                storage_exists = storage_or_static_path.exists(filename)

            if not storage_exists or filename.rsplit('.', 1)[-1] in ignore_file_type:
                logger.debug(f"loading {filename}")
                
                # kwargs will a contain a str `df_engine` on ^2.3.x
                return footprint_class(storage_or_static_path, *args, **kwargs)
    
    # If no valid footprint class was found, raise an error
    raise OasisFootPrintError(message="no valid footprint found")


#! PATCH THE FOOTPRINT LOAD METHOD
Footprint.load = classmethod(load)



###### FOOTPRINT SPECIFICATION
# currently the footprint is a sorted array of APIDs in a struct
# with the intensity 
# #   APID  : uint64
# #   intensity : uint32
# We have newer specifaction proposal as before with QK but for 
# the time being we are sticking with this. This array of structs
# for each event is zipped using zlib in the final footprint.
# As of current there is one feature that we maintain but may remove 
# in the near future and that is the stroing of APID with a null 
# intensity. This is required for the original and simpler 
# ZIGGURAT SEARCH but is not required of the newer ZIGGURAT TREE     

ZCBinEvent = nb.from_dtype(np.dtype([
    ('areaperil_id',     np.uint64),
    ('intensity_bin_id', np.int32)
]))

level_dtype = nb.from_dtype(np.dtype([
    ('size',     np.uint32), 
    ('leaves',   np.uint32), 
    ('capacity', np.uint32),
    ('offset',   np.uint32)
]))


parent_rec = nb.from_dtype(np.dtype([
    ('parent_idx',    np.int32),
    ('parent_offset', np.int32)
]))

# ! 2. ZIGGURAT TREE
# ! SCAN FOOTPRINT
# because we have the footprint sorted by level we can 
# effectively use this to make sure that each level is 
# only allocated once and we can use the max children 
# of the previous levels to calculate this
@nb.njit(cache=True)
def scan_footprint(footprint_arr, max_level):

    # since we are storing the root 0 node we need one extra level
    current_level: uint64 = uint64(0)
    subperils   = np.zeros(NUM_SUBPERILS, dtype=boolean)
    level_idx   = np.zeros(max_level + 1, dtype=uint32)
    level_size  = np.zeros(max_level + 1, dtype=uint32)
    levels      = np.zeros(max_level + 1, dtype=level_dtype)


    # we loop throught the footprint once to get the subperils
    # and the index of each level break
    for i in range(footprint_arr.shape[0]):
        apid: uint64 = footprint_arr[i]['areaperil_id']
        subperils[apid & 15] = True
        
        # we can skip when we are at the max level
        # since we know the max siae of the final level
        if current_level == max_level:
            continue

        level: uint64 = get_level(apid)
        # if a new level break we need to allocate the levels array in between
        if level != current_level:
            for lvl in range(current_level, level):
                level_idx[lvl] = i
            current_level = level
    # since we are tracking back we need to set the last level
    level_idx[current_level] = footprint_arr.shape[0]

    # now we can loop through the level breaks and get the size of each level
    for i in range(level_idx.shape[0]):
        # if basically there is only one qk of level 0
        # we need to at least set this
        if i == 0:
            level_size[i] = level_idx[i]
        else:
            # if the level is 0 and we know we arent level 0
            # then we know that we have hit the ends of the
            # levels present in the footprint
            if level_idx[i] == 0:
                continue
            level_size[i] = level_idx[i] - level_idx[i-1]


    # effectively calculates the maximum number of quadkeys at a level
    # given the previous levels and the number of children at that level
    # required for the effective contigious sotring of memory the alternative to this
    # is to loop through the footprint and count the number of quadkeys at each level
    num_qks           = 0 
    accumulated_leafs = 0
    for level in range(0, max_level + 1):
        max_qks = (1 << (level * 2))
        free_qks = max_qks - accumulated_leafs

        levels[level]['capacity'] = free_qks
        levels[level]['offset']   = num_qks

        num_qks += free_qks
        accumulated_leafs = (accumulated_leafs + level_size[level]) << 2

    return levels, subperils, num_qks


# ! GENERATE ZIGGURAT
# this function now goes through the footprint again with the 
# information from the scan and the initialized array and we 
# start to populate either the nodes or the leaves of the tree
# ? NOTE: the values in the quadtree are int32 and use sentinel
# ? values to determine if the node is a leaf or not 
# ? VALUES <  -1 are leaves and the value is the hazard id
# ? VALUES == -1 are uninitialized 
# ? VALUES >  -1 are nodes and the value is the offset to the next level
# This method also uses the z ordering of the quadkey in the items to 
# speed up the process of finding the leaf node (we know that if the 
# previous quadkey is initialized then the current quadkey and its parents
# are also initialized). 
@nb.njit(cache=True)
def generate_ziggurat(footprint_arr, levels, quadtree_arr, max_level):

    # momoization of the previous quadkey and it parent route
    prev_parent_idx = np.zeros(max_level + 1, dtype=parent_rec)
    for i in range(prev_parent_idx.shape[0]):
        prev_parent_idx[i]['parent_idx'] = -1
        prev_parent_idx[i]['parent_offset'] = -1

    prev_qk:    uint64 = np.uint64(0)
    prev_level: uint64 = np.uint64(0) 
    level_diff: uint64 = np.uint64(0)   
    shift_qk:   uint64 = np.uint64(0) 
    qk:         uint64 = np.uint64(0) 
    qk_match:   uint64 = np.uint64(0) 

    for i in range(footprint_arr.shape[0]):
        record:     uint64 = footprint_arr[i]
        apid:       uint64 = record['areaperil_id']
        apid_level: uint64 = get_level(apid)
        
        # qk = np.uint64(APID130ToQK(apid))
        qk: uint64 = (apid >> 4) & ((1 << (apid_level << 1)) - 1)

        # shift to largere  qk to the smaller qk and since the fp it sorted
        # by level we can assume that prev qk is always smaller
        # level_diff = np.uint64(apid_level - prev_level)
        level_diff = apid_level - prev_level
        # shift_qk   = np.uint64(qk) >> np.uint64(np.uint64(level_diff) << np.uint64(1)) 
        shift_qk   = qk >> (level_diff << 1) 
        qk_match   = shift_qk ^ np.uint64(prev_qk) 

        # now we can get the level of the last matching qk level
        match_level:   uint64 = np.uint64(0)
        parent_idx:    uint64 = np.uint64(0)
        parent_offset: uint64 = np.uint64(0)
        idx:           uint64 = np.uint64(0)
        if qk_match:
            match_level = prev_level - ((get_leading_bit_count(qk_match) >> 1))
            parent_offset = prev_parent_idx[match_level]['parent_offset']

        # with the updated matched level and the new parent offset
        # we can now skip through a bunch of looping which saves cpu 
        # cycles and memory access is tighter for the cache
        for level in range(match_level, apid_level):

            # get the idx of the record at the level
            # parent_idx = qk >> np.uint64((np.uint64(apid_level - level) << np.uint64(1))) & np.uint64(0b11)
            parent_idx = qk >> ((apid_level - level) << 1) & 0b11
            idx: uint64 = np.uint64(parent_offset + parent_idx)
            parent = quadtree_arr[idx] 

            # check to see if the parent level exists and if it doesnt create
            # forward childern at the next level and asing the offset
            # ? note that it should not be possible for a hazard to be set here
            # ? ie parent < -1 but we dont check because we can assume the i
            # ? is correct incoming from the user           
            if parent == -1: # no child for parent yet
                quadtree_arr[idx] = levels[level+1]['offset'] + levels[level+1]['size'] * 4 
                levels[level+1]['size'] += 1

            # parent_idx = parent 
            prev_parent_idx[level]['parent_idx']    = parent_idx
            prev_parent_idx[level]['parent_offset'] = parent_offset
            parent_offset = quadtree_arr[idx]
        

        # this is the final level and we can set the value of the record
        # at the leaf and move on
        # parent_idx = qk & np.uint64(0b11)
        parent_idx = qk & 0b11
        prev_parent_idx[apid_level]['parent_idx']    = parent_idx
        prev_parent_idx[apid_level]['parent_offset'] = parent_offset

        # ! this is where the hazard is transformed into the sentinel value
        quadtree_arr[np.uint64(parent_offset + parent_idx)] = -1 - record['intensity_bin_id']
        levels[apid_level]['leaves'] += 1

        prev_qk     = qk
        prev_level  = apid_level

    return quadtree_arr, levels




# ! REFACTOR FOOTPRINT
# this basically mimics the interface of the original footprint search method
# so we can easily swap between the two methods for testing and benchmarking
# ? this method can be speed up (potentially) by using the same memoization
# ? as the generate ziggurat method but thi current implementation is actually
# ? faster. Not sure but i assume it has to do with the distribution of the 
# ? locations (most are in the first few levels) and if the locations are 
# ? more concentrated and are sorted then the method may be fasted but given 
# ? the current implementation is as fast as it is i dont think its worth the effort
@nb.njit(cache=True)
def refactor_footprint_ziggurat(ziggurat, subperils, items):
    # create a new array to hold the results
    new_fp = np.empty(items.shape[0], dtype=Event)

    for i in range(items.shape[0]):
        new_fp[i]['areaperil_id']   = items[i]['areaperil_id']    
        new_fp[i]['probability']    = 1.0
        apid: uint64                = items[i]['areaperil_id'] 

        max_level:  uint64 = get_level(apid)
        qk:         uint64 = (apid >> 4) & ((1 << (max_level << 1)) - 1)
        subperil:   uint64 = apid & 15

        if (subperils[subperil] == False):
            new_fp[i]['intensity_bin_id'] = 1
    
        parent_offset:  uint64 = 0
        idx:            uint64 = 0
        parent_idx:     uint64 = 0
        for level in range(max_level+1):
            parent_idx = qk >> ((max_level - level) << 1) & 0b11
            idx = parent_offset + parent_idx
            parent = ziggurat[idx] 
            
            # check to see if the parent level exists
            # since now we are using the levels arrays using 0 as the indicator for 
            # the parent level we can use this to check if the parent exists using the
            # a negative value for the child pointer               
            if parent < -1: # no child get value
                new_fp[i]['intensity_bin_id'] = -(parent + 1) # convert to positive value
                break
            parent_offset = ziggurat[idx]
    return new_fp


# ! REFACTOR FOOTPRINT TREE
# This function was made to make it easier to debug and time the code
# also allocating memory using np rather than is some what faster so 
# is a nice side effect of the way this function is split
def refactor_footprint_tree(footprint, items):
    levels, subperils, num_qks = scan_footprint(footprint, NUM_LEVELS)

    quadtree_arr = np.full(num_qks, -1, dtype=np.int32)

    generate_ziggurat(footprint, levels, quadtree_arr, NUM_LEVELS)
    # footprint = refactor_footprint_ziggurat(quadtree_arr, subperils, items)
    return refactor_footprint_ziggurat(quadtree_arr, subperils, items)


class CombusZCBIN(Footprint):
    # needed for the super.load function
    footprint_filenames = [zcbin_filename, zcbin_index]

    def __enter__(self):
        # ! SET FOOTPRINT
        zcbin_file = self.stack.enter_context(self.storage.with_fileno(zcbin_filename))
        self.fp = mmap.mmap(zcbin_file.fileno(), length=0, access=mmap.ACCESS_READ)
        self.filesize = self.fp.size()

        footprint_header = np.frombuffer(bytearray(self.fp[:FootprintHeader.size]), dtype=FootprintHeader)
        self.num_intensity_bins = int(footprint_header['num_intensity_bins'])
        self.has_intensity_uncertainty = int(footprint_header['has_intensity_uncertainty'] & intensityMask)

        # ! SET INDEX
        zcbin_idx = self.stack.enter_context(self.storage.with_fileno(zcbin_index))
        footprint_mmap = np.memmap(zcbin_idx, dtype=EventIndexBin, mode='r')
        self.footprint_index = pd.DataFrame(footprint_mmap, columns=footprint_mmap.dtype.names)

        # remember that the bins use the abin file format so they actually are not in order
        self.footprint_index = self.footprint_index.set_index('event_id').to_dict('index')
        self.footprint_key_idx  = list(self.footprint_index)
        self.footprint_idx_map = {id:int(idx) for idx, id in enumerate(self.footprint_key_idx)}

        # ! READ ITEMS
        # this is the default it seems for the oasislmf runs since the call to footprint comes from the run dir
        run_dir = '.'  
        items_path = os.path.join(run_dir, 'input/items.bin')
        self.items = np.unique(np.memmap(items_path, dtype=Item, mode='r')['areaperil_id'])\
                                .astype(dtype=np.dtype([('areaperil_id', np.uint64)]))
        return self

    def unpack_event(self, event_id: int):
        event_info = self.footprint_index.get(event_id)
        event_idx = self.footprint_idx_map.get(event_id)
        if event_info is None:
            return 
        else:
            if event_idx + 1 != len(self.footprint_key_idx):
                next_offset = self.footprint_index.get(self.footprint_key_idx[event_idx + 1])['offset']
            else:
                next_offset = self.filesize

            zdata = self.fp[event_info['offset']: next_offset]
            data = zlib_ng.decompress(zdata)
            self.footprint = np.frombuffer(data, ZCBinEvent)

        
    def get_event(self, event_id):
        self.unpack_event(event_id) 
        if self.footprint is None:
            return None
        else:
            footprint = refactor_footprint_tree(self.footprint, self.items)  
            return footprint
 