import kagglehub

# Download latest version
path = kagglehub.dataset_download("hylanj/mini-imagenetformat-csv")

print("Path to dataset files:", path)