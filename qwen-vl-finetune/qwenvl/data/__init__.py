import re

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

VIDEOPROXY_DESC_SMOKE = {
    "annotation_path": "/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation_v1/qwen_sft_data/videoproxy_description_smoke.jsonl",
    "data_path": "",
}

VIDEOPROXY_DESC_10K = {
    "annotation_path": "/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation_v1/qwen_sft_data/videoproxy_description_10k.jsonl",
    "data_path": "",
}

VIDEOPROXY_DVC_10K = {
    "annotation_path": "/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation_v1/qwen_sft_data/videoproxy_dense_video_caption_10k.jsonl",
    "data_path": "",
}

VIDEOPROXY_PROXY_MIX_SFT = {
    "annotation_path": "/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/multi_task/experiments/composition_base_seg_logic_aot_hier10k_el10k_aot10k_mf256_ema/qwen_sft/proxy_mix_train_sft.jsonl",
    "data_path": "",
}

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    "videoproxy_desc_smoke": VIDEOPROXY_DESC_SMOKE,
    "videoproxy_desc_10k": VIDEOPROXY_DESC_10K,
    "videoproxy_dvc_10k": VIDEOPROXY_DVC_10K,
    "videoproxy_proxy_mix_sft": VIDEOPROXY_PROXY_MIX_SFT,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
