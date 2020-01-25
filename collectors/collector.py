"""A super class for data collectors. Typical collectors include daily newspaper article collectors"""
from attr import attrs, attrib

from commons.datamodel import DataModel


@attrs
class Collector:
    frequency = attrib(default=86400)  # seconds to sleep before running the collection cycle
    data = attrib(default=DataModel())  # DataModel representation of the collected data
    download_fresh = attrib(default="yes")  # If download_fresh is false, use previously downloaded data.

    def run(self, loop):
        pass

    def gather_data(self, loop):
        pass
