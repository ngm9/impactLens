"The NYT collector of everyday's articles"
import logging
import os
import configparser
import requests
import pyjq
import asyncio
from newspaper import Article
import datetime
import attr

from collectors.collector import Collector
from commons.datamodel import DataModel
from attr import attrib

logger = logging.getLogger(__name__)

NYT_SECTION = "NYT"


@attr.s
class NytCollector(Collector):
    key = attrib(default="")
    url = attrib(default="")
    datapath = attrib(default=".")
    last_refresh = attrib(default=datetime.datetime.now())

    def run(self, loop):
        asyncio.set_event_loop(loop)
        logger.info(f"current loop time: {loop.time()}. Scheduling data gathering callback for {self.frequency}")
        loop.call_soon(self.gather_data, loop)
        try:
            loop.run_forever()
        finally:
            loop.close()

    def build_config(self, config_file="./config/prod.ini"):
        cparser = configparser.ConfigParser()
        cparser.read(config_file)

        for section in cparser.sections():
            if section == NYT_SECTION:
                self.key = cparser[section]['key']
                self.url = cparser[section].get('url')
                self.datapath = cparser[section]['datapath']
                self.frequency = int(cparser[section]['frequency'])
                self.download_fresh = cparser[section].getboolean('download_fresh')
                break
        logger.info(f"Loaded configuration for NYT collector: {self}")

    def gather_data(self, loop):
        dataArr = []
        count = 0
        logger.info(f"Entering gather_data. download_fresh: {self.download_fresh}")
        if not self.download_fresh:
            logger.info(f"Current time is: {datetime.datetime.now().strftime('%c')}")
            logger.info("Building Dataset Building test dataset from NYT's archive of 12/2018.")
            logger.info(f"Loading articles from f{self.config.datapath}")
            files = os.listdir(self.config.datapath)
            files = [os.path.join(self.config.datapath, f) for f in files]
            files.sort(key=lambda x: os.path.getmtime(x))
            for file in files:
                with open(file, "r") as f:
                    dataArr.append(f.read())
                    count += 1
        else:
            logger.info(f"Current time is: {datetime.datetime.now().strftime('%c')}")
            logger.info("Building Dataset from NYT's archive of 12/2018. Downloading ...")

            url = self.url + "?&api-key=" + self.key
            logger.info(f"url: {url}")
            r = requests.get(url)
            logger.debug(f"Downloaded {len(r.text)} bytes of data")
            json_data = r.json()

            jq = f".response .docs [] | .web_url"
            out = pyjq.all(jq, json_data)

            # some URLs are empty, clean up
            cleanUrls = self.cleanup_urls(input_urls=out)
            self.last_refresh = datetime.datetime.now()
            self.build_datamodel(data=cleanUrls)
            loop.call_later(self.frequency, self.gather_data, loop)

    def build_datamodel(self, data) -> DataModel:
        count = 0
        dataArr = []

        logger.info(f"Attempting to download articles from individual links")
        for url in data:
            if count > 100:  # if you don't want to download 6200 articles
                break
            a = Article(url=url)
            try:
                a.download()
                a.parse()
            except Exception as ex:
                print(f"caught {ex} continuing")
            if len(a.text):
                print(f"{len(dataArr)} - downloaded {len(a.text)} bytes")
                dataArr.append(a.text)
                with open(self.datapath + "/" + f"{count}.txt", "w") as f:
                    f.write(a.text)
                count += 1

        logger.info(f"working with {len(dataArr)} articles")
        self.data = DataModel()
        self.data.setData(dataArr)

    def cleanup_urls(self, input_urls):
        cleanUrls = []
        for url in input_urls:
            if not url:  # some URLs are empty
                continue
            cleanUrls.append(url)
        logger.info(f"Clean url count: {len(cleanUrls)}")
        return cleanUrls

