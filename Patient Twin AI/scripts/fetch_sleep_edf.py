import concurrent.futures
import os
import re
import urllib.request

BASE_URL = "https://physionet.org/files/sleep-edfx/1.0.0/"
DEST_DIR = "datasets/Sleep-EDF"


def get_links(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        html = urllib.request.urlopen(req, timeout=10).read().decode('utf-8')
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return []
    links = re.findall(r'href="([^"]+)"', html)
    return [link for link in links if not link.startswith('?') and link not in ('../', '/')]


def download_file(url, dest):
    tmp_dest = dest + ".tmp"
    req = urllib.request.Request(
        url, headers={'User-Agent': 'Mozilla/5.0', 'Connection': 'close'}
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            remote_size = int(response.getheader('Content-Length', -1))
    except Exception as e:
        print(f"Error checking size for {url}: {e}")
        return

    if os.path.exists(dest):
        if remote_size != -1 and os.path.getsize(dest) == remote_size:
            return
        else:
            os.remove(dest)

    print(f"Downloading {dest}...")
    try:
        with (
            urllib.request.urlopen(req, timeout=30) as response,
            open(tmp_dest, 'wb') as out_file,
        ):
            while True:
                chunk = response.read(8192 * 8)
                if not chunk:
                    break
                out_file.write(chunk)
        if remote_size != -1 and os.path.getsize(tmp_dest) != remote_size:
            raise Exception("Size mismatch")
        os.rename(tmp_dest, dest)
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(tmp_dest):
            os.remove(tmp_dest)


def main():
    os.makedirs(DEST_DIR, exist_ok=True)
    files_to_download = []

    for link in get_links(BASE_URL):
        if link.endswith('/'):
            subdir_url = BASE_URL + link
            subdir_dest = os.path.join(DEST_DIR, link.strip('/'))
            os.makedirs(subdir_dest, exist_ok=True)
            for sub_link in get_links(subdir_url):
                if not sub_link.endswith('/'):
                    files_to_download.append(
                        (subdir_url + sub_link, os.path.join(subdir_dest, sub_link))
                    )
        else:
            files_to_download.append((BASE_URL + link, os.path.join(DEST_DIR, link)))

    print(f"Found {len(files_to_download)} files. Checking and fixing concurrently...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(download_file, url, dest) for url, dest in files_to_download]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    print("Sleep-EDF parallel download complete.")


if __name__ == "__main__":
    main()
