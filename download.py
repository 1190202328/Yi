# 模型下载
from modelscope import snapshot_download

model_dir = snapshot_download(
    model_id='01ai/Yi-VL-34B',
    cache_dir='/nfs/ofs-902-vlm/jiangjing/',
    allow_file_pattern=['*'],
)
