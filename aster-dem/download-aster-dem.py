import os
import requests

def download(url: str, dest_folder: str):
    #https://stackoverflow.com/a/56951135/8761164
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    filename = url.split('/')[-1].replace(" ", "_")  # be careful with file names
    file_path = os.path.join(dest_folder, filename)

    r = requests.get(url, stream=True)

    if r.ok:
        print("saving to", os.path.abspath(file_path))
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else:
        print("Download failed: status code {}\n{}".format(r.status_code, r.text))


def parse_lat(lat: int):
    lat_str = 'N' if lat >= 0 else 'S'
    if 10 > lat > -10:
        lat_str += '0'
    lat_str += str(abs(lat))
    return lat_str

def parse_long(long: int):
    long_str = 'E' if long >= 0 else 'W'
    if 100 > long > -100:
        long_str += '0'
    if 10 > long > -10:
        long_str += '0'
    long_str += str(abs(long))
    return long_str


if __name__=='__main__':

    for lat in range(47, 21, -1):
        for long in range(-14, 43, 1):
            #print(parse_lat(lat), parse_long(long))
            #print(f"https://gdemdl.aster.jspacesystems.or.jp/download/Download_{parse_lat(lat)}{parse_long(long)}.zip")
            download(f"https://gdemdl.aster.jspacesystems.or.jp/download/Download_{parse_lat(lat)}{parse_long(long)}.zip", dest_folder="/media/data-ext/aster-gdem")