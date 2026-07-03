import urllib.request
import re
import os
import concurrent.futures

BASE_URL = "https://physionet.org/files/sleep-edfx/1.0.0/"
DEST_DIR = "datasets/Sleep-EDF"

def get_links(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        html = urllib.request.urlopen(req).read().decode('utf-8')
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return []
    links = re.findall(r'href="([^"]+)"', html)
    # Filter out parent dir and query params
    return [l for l in links if not l.startswith('?') and l != '../' and l != '/']

def download_file(url, dest):
    if os.path.exists(dest):
        return
    print(f"Downloading {dest}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response, open(dest, 'wb') as out_file:
            while True:
                chunk = response.read(8192 * 8)
                if not chunk:
                    break
                out_file.write(chunk)
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def main():
    os.makedirs(DEST_DIR, exist_ok=True)
    files_to_download = []
    
    # Root files
    for link in get_links(BASE_URL):
        if link.endswith('/'):
            # Subdirectory
            subdir_url = BASE_URL + link
            subdir_dest = os.path.join(DEST_DIR, link.strip('/'))
            os.makedirs(subdir_dest, exist_ok=True)
            for sub_link in get_links(subdir_url):
                if not sub_link.endswith('/'):
                    files_to_download.append((subdir_url + sub_link, os.path.join(subdir_dest, sub_link)))
        else:
            files_to_download.append((BASE_URL + link, os.path.join(DEST_DIR, link)))

    print(f"Found {len(files_to_download)} files. Downloading concurrently...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(download_file, url, dest) for url, dest in files_to_download]
        for future in concurrent.futures.as_completed(futures):
            future.result()
            
    print("Sleep-EDF parallel download complete.")

if __name__ == "__main__":
    main()
