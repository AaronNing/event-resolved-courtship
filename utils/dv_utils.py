import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from dv import AedatFile
import pandas as pd



def getPacket(it: AedatFile.numpy_packet_iterator) -> tuple:

    try:
        packet = pd.DataFrame(next(it)).iloc[:,:4]
    except StopIteration:
        return False, None
    else:
        return True, packet



def getFirstTs(packet: pd.DataFrame) -> np.int64:
    
    return packet['timestamp'].min()



def filterPacket(packet: pd.DataFrame, roi, pol: str='positive') -> pd.DataFrame:

    packet = packet[packet['x'] > roi[0]]
    packet = packet[packet['x'] < roi[0] + roi[2]]
    packet = packet[packet['y'] > roi[1]]
    packet = packet[packet['y'] < roi[1] + roi[3]]

    if (pol == 'positive'):
        polarity = 1
    else:
        polarity = 0

    packet = packet[packet['polarity'] == polarity]

    return packet



def packetContainBin(packet: pd.DataFrame, bin) -> bool:

    if packet['timestamp'].max() > bin[1]:
        return True
    else:
        return False



def combinePackets(p1: pd.DataFrame, p2: pd.DataFrame) -> pd.DataFrame: 

    return pd.concat([p1, p2], axis=0)




def cutBinFromPacket(packet: pd.DataFrame, bin) -> tuple:
    
    in_bin_bool = packet['timestamp'] < bin[1]

    return packet[~in_bin_bool], in_bin_bool.sum()




def accumulate(aedat_path: str, fps: int, roi, tw_mode=False, tw=[0,1]) -> np.ndarray:
    '''
    roi: cv-like roi. 

    when tw_mode == False, tw is ignored. 
    '''
    assert(isinstance(fps, int))
    assert(1e6 % fps == 0)
    binsize = 1e6/fps

    out = []
    bin_cnt = 0
    with AedatFile(aedat_path) as f:
        it = f['events'].numpy()

        success, first_packet = getPacket(it)
        if not success:
            raise(Exception("failed reading first packet!"))
        first_ts = getFirstTs(first_packet)
            
        first_packet = filterPacket(first_packet, roi=roi, pol='positive')
        packet_on_hand = first_packet

        bin = [0, binsize] + first_ts

        # tw mode
        if (tw_mode):
            bin += tw[0] * 1e6

            def not_reaching_tw(packet, tw):
                return packet['timestamp'].max() < first_ts + 1e6 * tw[0]

            def passed_tw(packet, tw):
                return packet['timestamp'].min() > first_ts + 1e6 * tw[1]


        packet_cnt = 1
        while(True): # bin

            while(not packetContainBin(packet_on_hand, bin)):
                success, packet = getPacket(it)
                if not success:
                    break
                else:
                    packet_cnt += 1
                    if (packet_cnt % 100 == 0):
                        print("passing packet [" + str(packet_cnt) + "]")

                    # tw mode
                    if (tw_mode):
                        if not_reaching_tw(packet, tw):
                            continue
                        elif passed_tw(packet, tw):
                            break

                    packet = filterPacket(packet, roi=roi, pol='positive')
                    packet_on_hand = combinePackets(packet_on_hand, packet)

            if not success:
                break

            packet_on_hand, bin_sum = cutBinFromPacket(packet_on_hand, bin)
            
            out.append(bin_sum)

            #
            bin += binsize

            # tw mode
            if (tw[1] * 1e6 + first_ts < bin[0]):
                break

            bin_cnt += 1
            if (bin_cnt % fps == 0):
                print('Finishing bin [' + str(bin_cnt) + '], time [' + str(bin_cnt/fps) + ']s')
            
    return np.array(out[1:])