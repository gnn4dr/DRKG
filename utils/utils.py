import os
import tarfile

def download_and_extract():
    import shutil
    import requests
    
    url = "https://s3.us-west-2.amazonaws.com/dgl-data/dataset/DRKG/drkg.tar.gz"
    path = "../data/"
    filename = "drkg.tar.gz"
    if os.path.exists(os.path.join(path, filename)):
        return
    
    opener, mode = tarfile.open, 'r:gz'
    os.makedirs(path, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(path)
    while True:
        try:
            with opener(filename, mode) as file:
                file.extractall()
            break
        except Exception:
            f_remote = requests.get(url, stream=True)
            sz = f_remote.headers.get('content-length')
            assert f_remote.status_code == 200, 'fail to open {}'.format(url)
            with open(filename, 'wb') as writer:
                for chunk in f_remote.iter_content(chunk_size=1024*1024):
                    writer.write(chunk)
            print('Download finished. Unzipping the file...')
    os.chdir(cwd)
