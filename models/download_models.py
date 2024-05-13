from huggingface_hub import snapshot_download
from huggingface_hub import HfApi


api = HfApi()

repo = "IEETA/Multi-Head-CRF"

refs = api.list_repo_refs(repo)
for branch in refs.branches:
    name = branch.name
    if name!= 'main':
        snapshot_download(repo_id=repo, revision=name, local_dir = './'+name)

