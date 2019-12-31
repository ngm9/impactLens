"""Utils to work with New York Times data."""

from commons.datamodel import DataModel
import os
import requests
import pyjq
from newspaper import Article


def cleanUpURLs(out):
    cleanUrls = []
    for url in out:
        if not url:  # some URLs are empty
            continue
        cleanUrls.append(url)
    return cleanUrls


def buildTestDataFromNYT(download=True, articlesDir=".", writeToDir=False):
    testData = DataModel()
    dataArr = []
    count = 0

    if not download:
        print("\nStep 1: Building Test Dataset Building test dataset from NYT's archive of 12/2018.")
        print(f"Loading articles from f{articlesDir}")
        files = os.listdir(articlesDir)
        files = [os.path.join(articlesDir, f) for f in files]
        files.sort(key=lambda x: os.path.getmtime(x))
        for file in files:
            with open(file, "r") as f:
                dataArr.append(f.read())
                count += 1
    else:
        print("Building test dataset from NYT's archive of 12/2018. Downloading ...")
        key = "5AE0mAEtH2uXTpUjUnNr4kS9GVTVco8M"

        url = 'https://api.nytimes.com/svc/archive/v1/2018/12.json?&api-key=' + key
        r = requests.get(url)
        json_data = r.json()

        jq = f".response .docs [] | .web_url"
        out = pyjq.all(jq, json_data)

        # some URLs are empty, clean up
        cleanUrls = cleanUpURLs(out)

        for url in cleanUrls:
            # if count > 3000:  # if you don't want to download 6200 articles
            #     break
            a = Article(url=url)
            try:
                a.download()
                a.parse()
            except Exception as ex:
                print(f"caught {ex} continuing")
            if len(a.text):
                print(f"{len(dataArr)} - downloaded {len(a.text)} bytes")
                dataArr.append(a.text)
                if writeToDir:
                    with open(articlesDir + "/" + f"{count}.txt", "w") as f:
                        f.write(a.text)
                count += 1

    print(f"working with {len(dataArr)} articles")
    testData.setDocuments(dataArr)
    return testData
