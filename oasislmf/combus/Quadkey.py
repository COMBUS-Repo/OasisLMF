import numba as nb
from numba.types import uint32, uint64, float64

##### SPECIFICATION FOR APID
# Currently our specification for out APID is as follows
# uint64 with a leading one to preserve level and 4 bit reserved
#  at the end for the subperil... while i would like very much to 
# store the level in the  APID the specification preceeded much 
# optimisation and and being integral to most of our other code 
# it has remeained the case. Beside the operation we have optimised 
# to get the level is next to insignificant for runtime costs.
# (alot of our optimisations take advatage of AVX/SIMD as well as
# ASM and BMI2 instrinics for the interleaving and deinterleaving of 
# bits - unsure of portability with python or numba?)
#
# Perhaps the quadkey APID specification could be left
# as a potential user implementation? Allowing for use case specific 
# optimisations ?? Note that we at combus have a next version 
# specification proposal but no implmenation has been made. 
# 
# As for the QK the bit pairs for the quakdey are stored 
# in increasing order for example QK that would correspond
#  with a QK string of 130 (level 3) and subperil 5
#       0.....1  011100  0101
#               |_1_3_0|
#                      \           
# NOTE: in out nomenclature QK refers to this part and APID
# refers to the whole bit string 
# 
# There is some justification ofr having the QK placed in this
# but most important is that is 
# #    IT ENSURES THAT QKS ARE SIMPLY SORTED IN A Z-ORDER
# #    CURVE BY LEVEL FIRST
# this is important to us for performance but also many of our 
# algorithms we use for footprints rely on this property
# in your sepcification of QK this can be done by placing the 
# level bytes before the QK bytes in the APID


# These are essentially arbitrary and can be user defined 
# also ignore the 130 in the QK130 part it is a again a 
# remanant of an earlier specification that allows us to in the future
# increase the size of our originating QK to allowfor modelling 
# other countries such as NZ etc 
# NOTE: this QK was defined around the idea that our final level 
# would be one arcsecond (1/3600 of a degree) thus the weird definition
# it being in reverse rather than interior levels being defined by 
# the outer
QK130_TL_LON = float64(-330.0 + (pow(2.0,20) + pow(2.0,19)) / 3600.0 - 1.0 / 7200.0)
QK130_TL_LAT = float64( 145.0 - (pow(2.0,19))               / 3600.0 + 1.0 / 7200.0)
QK130_BR_LON = float64(QK130_TL_LON + (pow(2.0,18) / 3600.0))
QK130_BR_LAT = float64(QK130_TL_LAT - (pow(2.0,18) / 3600.0))


# so one of the neat things about quadkey which you would have noticed 
# from writing the original loops for the generation of the QK 
# is that the quadkey is a simple interleaving of the x and y index
# leveraging this using bit mask operations the operation to 
# generate a quadkey and its inverse is extremely fast
#
# NOTE: that we try to isolate the QK operation from the operations
# on a APID and we convert between QK and APID in our C++ 
# implementation. This is to allows for a cleaner abstraction for
# us as we use some other properties of quadkeys in other 
# algorithms but for the purpose of the converting between 
# coordinates and APIDs (the only thing we do with python we 
# have shortcut some of these represenatationl decisions)
@nb.njit(uint64(uint32, uint32))
def interleave_bits(x: uint32, y: uint32) -> uint64:
    # explode bits of x
    x64: uint64 = 0
    x64 |= (x & 0x0000FFFF) | ((x & 0xFFFF0000) << 16)
    x64 = (x64 | (x64 << 8)) & 0x00FF00FF00FF00FF
    x64 = (x64 | (x64 << 4)) & 0x0F0F0F0F0F0F0F0F
    x64 = (x64 | (x64 << 2)) & 0x3333333333333333
    x64 = (x64 | (x64 << 1)) & 0x5555555555555555

    # explode bits of y
    y64: uint64 = 0
    y64 |= (y & 0x0000FFFF) | ((y & 0xFFFF0000) << 16)
    y64 = (y64 | (y64 << 8)) & 0x00FF00FF00FF00FF
    y64 = (y64 | (y64 << 4)) & 0x0F0F0F0F0F0F0F0F
    y64 = (y64 | (y64 << 2)) & 0x3333333333333333
    y64 = (y64 | (y64 << 1)) & 0x5555555555555555

    # Combine interleaved bits of x and y
    return (y64 << 1) | x64

# the inverse of the interleave bits function
# this needs to be called twice for a QK 
# once to automatically get the X-index 
# and thaking the QK >> 1 to get the Y-index
@nb.njit(uint32(uint64))
def collapse_bits(bits: uint64) -> uint32:
    # Mask to keep only even bits
    even_mask: uint64 = (0x5555555555555555)
    bits = bits & even_mask

    # Collapse even bits into a 32-bit integer
    bits = (bits | (bits >>  1)) & (0x3333333333333333)  # Combine pairs
    bits = (bits | (bits >>  2)) & (0x0F0F0F0F0F0F0F0F)  # Combine nibbles
    bits = (bits | (bits >>  4)) & (0x00FF00FF00FF00FF)  # Combine bytes
    bits = (bits | (bits >>  8)) & (0x0000FFFF0000FFFF)  # Combine words
    bits = (bits | (bits >> 16)) & (0x00000000FFFFFFFF)  # Combine into int

    return uint32(bits)


# This is not as efficient as storing the level in the bit 
# unfortunately our original specificationdid not allow for
# for this but this is an effective way of calculating the
# level from the APID without much penalty at runtime
@nb.njit(uint64(uint64), inline='always')
def get_leading_bit_count(x: uint64) -> uint64:
    position: uint64 = 0
    if x >= (1 << 32):
        position += 32
        x >>= 32
    if x >= (1 << 16):
        position += 16
        x >>= 16
    if x >= (1 << 8):
        position += 8
        x >>= 8
    if x >= (1 << 4):
        position += 4
        x >>= 4
    if x >= (1 << 2):
        position += 2
        x >>= 2
    if x >= (1 << 1):
        position += 1

    return position


### for our QK spec we have calculate the level 
# from the leading bit count based on the spec
@nb.njit(uint64(uint64))
def get_level(quadkey_bin: uint64) -> uint64:
    # Get the leading bit count
    leading_bit_count = get_leading_bit_count(quadkey_bin)

    # Calculate the level based on the leading bit count
    level: uint64 = ((leading_bit_count - 4) >> 1)
    
    return level


# convesion tool to covert between the APID130 schema and the QK
@nb.njit(uint64(uint64))
def APID130ToQK(APID: uint64) -> uint64:
    # shift and remove the leading byte to get the QK
    level = get_level(APID)
    qk = (APID >> 4) & ((1 << (level << 1)) - 1)
    return qk 


# and it inverse
@nb.njit(uint64(uint64, uint64))
def QKToAPID130(qk: uint64, level: uint64) -> uint64:
    # add leading byte and ready for the subperil 
    LEADING_BYTE: uint64 = (1 << (level << 1)) 
    return (LEADING_BYTE ^ qk) << 4


@nb.njit(uint64(float64, float64, uint64))
def coordToAPID130(lon: float64, lat: float64, level: uint64) -> uint64:
        QKS_IN_DEGREE =  float64(3600.0) / float64(1 << 18 - level)
        x_index = uint32((lon - QK130_TL_LON) * QKS_IN_DEGREE)
        y_index = uint32((QK130_TL_LAT - lat) * QKS_IN_DEGREE)
        qk = uint64(interleave_bits(x_index, y_index))
        
        #add leading byte and ready for the subperil 
        LEADING_BYTE = 1 << (level << 1)
        return (LEADING_BYTE ^ qk) << uint64(4)

# Define a tuple type for coordinates (lon, lat)
Coord = nb.types.Tuple((nb.types.float64, nb.types.float64))
@nb.njit(Coord(uint64))
def APID130ToCoord(APID: uint64) -> Coord:
        # shift and remove the leading byte to get the QK
        level = get_level(APID)
        qk = (APID >> 4) & ((1 << (level << 1) - 1))

        QKS_SIZE = float64(1 << 18 - level) / float64(3600.0)
        x_index = collapse_bits(qk) 
        y_index = collapse_bits(qk >> 1)
         
        return (QK130_TL_LON + ((x_index + 0.5) * QKS_SIZE),
                QK130_TL_LAT - ((y_index + 0.5) * QKS_SIZE))

