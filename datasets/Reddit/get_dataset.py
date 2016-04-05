#!/usr/bin/env python3

import json

from time import sleep

from urllib.request import Request
from urllib.request import urlopen
from urllib.request import urlretrieve
from urllib.parse   import urlencode
from urllib.parse   import urlsplit

def get_dataset():
    urls = get_urls([
        'CityPorn',
        'EarthPorn',
        'photocritique',
        'WaterPorn',
        #'itookapicture',
        #'windowshots',
    ], 3000)

    i = 1
    for url in urls:
        success = get_photo(i, url)
        if success:
            i += 1


def get_urls(subreddits, n):

    api_url = 'https://api.reddit.com/r/%s/top' % '+'.join(subreddits)

    last_name = ''
    urls = []

    while len(urls) < n:
        options = urlencode({
            't':     'all',     # All time
            'after': last_name, # Previous result
            'limit': 100,       # Return 100 results (max)
        })

        # Try to get API response
        data = None
        try:
            response = urlopen('%s?%s' % (api_url, options))
            charset = response.headers.get_content_charset() or 'utf-8'
            data = json.loads(response.read().decode(charset))['data']['children']
            response.close()
        except:
            sleep(2) # Pause a little
            continue

        print('> Received %d entries.' % len(data))
        if len(data) == 0:
            break

        for child in data:
            url = child['data']['url']

            dom = child['data']['domain']

            # Downloads disabled for flickr.com links, avoid
            if dom == 'flickr.com' or dom.endswith('.minus.com'):
                continue

            # Fix URLs for imgur links
            if dom == 'imgur.com' or dom == 'm.imgur.com':
                url = 'http://i.imgur.com/%s.jpg' % url.split('/')[-1]

            urls.append(url)

        last_name = data[-1]['data']['name']

    return urls[:n]


def get_photo(i, url):
    o = urlsplit(url)

    # Choose output file name
    fn = ''
    if o.netloc == 'drscdn.500px.org': # Handle URLs without .jpg ending
        fn = '%04d.jpg' % i
    else:
        ext = o[2].split('.')[-1]
        fn  = '%04d.%s' % (i, ext)

    print('- [%04d] Getting %s' % (i, url))

    # Download URL to file specified by fn
    try:
        response = urlopen(Request(url, headers={'User-Agent': 'Auto-Crop Study'}))
        code = response.getcode()
        length = int(response.info()['Content-Length'])
        if length < 100000:
            raise Exception("Content-Length too short: %d" % length);
        if code != 200:
            raise Exception("Response Code %d" % code)
        with open(fn, 'wb') as f:
            f.write(response.read())
        response.close()
    except Exception as err:
        print(err)
        print('! [%04d] Failed.' % i)
        return False

    return True


if __name__ == '__main__':
    get_dataset()
