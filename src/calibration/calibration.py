import numpy as np
from xml.dom import minidom
from colorama import init, Fore, Style
import time

class calibration:
    def __init__(self, x=np.zeros(6)):
        # Leverarm parameters [m]
        self.tx = x[0]
        self.ty = x[1]
        self.tz = x[2]
        # Boresight angle parameters [°]
        self.rx = x[3]
        self.ry = x[4]
        self.rz = x[5]
    
    def read_calibration_from_xml( self, path ):
        
        t = time.strftime("%H:%M:%S")
        print(( f"| {Style.BRIGHT}{Fore.GREEN}{t + ' Read calibration file from folder'}{Style.RESET_ALL}" ))
        print("| - path: ", path)
        
        xml_dom = minidom.parse( path )
        self.tx = float (xml_dom.getElementsByTagName("x")[0].firstChild.nodeValue )
        self.ty = float (xml_dom.getElementsByTagName("y")[0].firstChild.nodeValue )
        self.tz = float (xml_dom.getElementsByTagName("z")[0].firstChild.nodeValue )
        self.rx = float (xml_dom.getElementsByTagName("rx")[0].firstChild.nodeValue )
        self.ry = float (xml_dom.getElementsByTagName("ry")[0].firstChild.nodeValue )
        self.rz = float (xml_dom.getElementsByTagName("rz")[0].firstChild.nodeValue )

        print("| - txyz: ", self.tx, self.ty, self.tz)
        print("| - rxyz: ", self.rx, self.ry, self.rz)
        print("|  ")