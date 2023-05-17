import gdown

url = "https://drive.google.com/uc?id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45"
output = "output_filename.tgz"  # Here you can choose the output file name
gdown.download(url, output, quiet=False)
