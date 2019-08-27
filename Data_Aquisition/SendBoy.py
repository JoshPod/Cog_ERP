"""Example program to demonstrate how to send string-valued markers into LSL."""

import random
import time

from pylsl import StreamInfo, StreamOutlet

# first create a new stream info (here we set the name to MyMarkerStream,
# the content-type to Markers, 1 channel, irregular sampling rate,
# and string-valued data) The last value would be the locally unique
# identifier for the stream as far as available, e.g.
# program-scriptname-subjectnumber (you could also omit it but interrupted
# connections wouldn't auto-recover). The important part is that the
# content-type is set to 'Markers', because then other programs will know how
#  to interpret the content

# Name / Type / Channel Count / Nominal SRate / Format / Stream ID.


info = StreamInfo('MyMarkerStream', 'Markers', 1, 0, 'string', 'send_boy')


# next make an outlet
outlet = StreamOutlet(info)

print("now sending markers...")
markernames = ['0', '1', '2', '3', '4', '5', '6']
while True:
    # pick a sample to send an wait for a bit
    outlet.push_sample([random.choice(markernames)])
    time_step = 1  # random.random() * 3
    time.sleep(time_step)
