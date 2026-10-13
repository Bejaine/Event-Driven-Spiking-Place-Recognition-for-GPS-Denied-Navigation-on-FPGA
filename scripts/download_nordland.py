"""Download Nordland (Somayeh-h/Nordland on HuggingFace) tarballs used by NRD_SFS."""
import sys
from huggingface_hub import hf_hub_download
for season in sys.argv[1:]:
    p = hf_hub_download("Somayeh-h/Nordland", f"data/{season}.tar.gz", repo_type="dataset",
                        local_dir="data/raw")
    print("downloaded", p, flush=True)
