import argparse
import json
import os

import torch
from PIL import Image

from llava.conversation import conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    expand2square,
    get_model_name_from_path,
    load_pretrained_model,
    tokenizer_image_token,
)
from llava.model.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, key_info


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def single_infer(args):
    split = 'NEW_Mini'
    # load model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    key_info["model_path"] = model_path
    get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path)
    model = model.to(dtype=torch.bfloat16)

    # fixed para
    answer_keys = {'question_id', 'image', 'question', 'answer'}
    save_dir = '/nfs/ofs-902-1/object-detection/jiangjing/experiments/Yi/VL/ans'
    save_dir = f'{save_dir}/{model_path.split("/")[-1]}/{split}'
    data_ann_root_dir = '/nfs/ofs-902-1/object-detection/tangwenbo/vlm/data/CODA-LM'
    data_img_root_dir = '/nfs/ofs-902-1/object-detection/tangwenbo/vlm/data/'
    data_ann_dir = f'{data_ann_root_dir}/{split}/vqa_anno'
    task_list = ['general_perception', 'region_perception', 'driving_suggestion']

    def get_ans(task_json, use_cache):
        image_file = f"{data_img_root_dir}/{task_json['image']}"
        qs = task_json['question']

        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        image = Image.open(image_file)
        if getattr(model.config, "image_aspect_ratio", None) == "pad":
            image = expand2square(
                image, tuple(int(x * 255) for x in image_processor.image_mean)
            )
        image_tensor = image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        stop_str = conv.sep
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).to(dtype=torch.bfloat16).cuda(),
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                stopping_criteria=[stopping_criteria],
                max_new_tokens=1024,
                use_cache=use_cache,
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
        outputs = tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()

        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        return outputs

    # start chat
    task_list_json = []
    for t in task_list:
        task_path = f'{data_ann_dir}/{t}.jsonl'
        with open(task_path, mode='r', encoding='utf-8') as f:
            task_list_json.append(f.read().strip().split('\n'))
    num_samples = len(task_list_json[0])

    answer_list = {
        'general_perception': [],
        'region_perception': [],
        'driving_suggestion': []
    }

    region_i = 0
    for i in range(num_samples):
        print(f'evaluating [{i + 1}/{num_samples}]')

        task_json = json.loads(task_list_json[0][i])
        ori_img_name = task_json['image'].split('/')[-1].split('.')[0]

        # 2. region_perception
        task_id, task_name = 1, 'region_perception'
        use_cache = False
        while True:
            try:
                task_json = json.loads(task_list_json[task_id][region_i])
            except IndexError:
                break
            img_name = task_json['image'].split('/')[-1].split('.')[0]
            if img_name.find(ori_img_name) == -1:
                # not found
                break

            region_i += 1
            answer = get_ans(task_json, use_cache=use_cache)
            use_cache = True
            # save_ans
            answer_list[task_name].append(answer)

        # 1. general_perception
        task_id, task_name = 0, 'general_perception'
        task_json = json.loads(task_list_json[task_id][i])

        answer = get_ans(task_json, use_cache=use_cache)
        # save_ans
        answer_list[task_name].append(answer)

        # 3. driving_suggestion
        task_id, task_name = 2, 'driving_suggestion'
        task_json = json.loads(task_list_json[task_id][i])

        answer = get_ans(task_json, use_cache=use_cache)
        # save_ans
        answer_list[task_name].append(answer)

    # dump ans
    task_list_json_dump = {
        'general_perception': [],
        'region_perception': [],
        'driving_suggestion': []
    }
    for task_id, task_name in enumerate(task_list):
        assert len(task_list_json[task_id]) == len(answer_list[task_name])

        for i in range(len(answer_list[task_name])):
            task_json = json.loads(task_list_json[task_id][i])
            task_json['answer'] = answer_list[task_name][i]

            # only answer keys are dumped
            new_task_json = {}
            for key in task_json:
                if key in answer_keys:
                    new_task_json[key] = task_json[key]
            task_list_json_dump[task_name].append(json.dumps(new_task_json))

    os.makedirs(save_dir, exist_ok=True)
    for task_name in task_list:
        save_path = f'{save_dir}/{task_name}_answer.jsonl'
        with open(save_path, mode='w', encoding='utf-8') as f:
            f.write('\n'.join(task_list_json_dump[task_name]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="01-ai/Yi-VL-6B")
    parser.add_argument("--image-file", type=str, default="images/cats.jpg")
    parser.add_argument(
        "--question",
        type=str,
        default="Describe the cats and what they are doing in detail.",
    )
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="mm_default")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    single_infer(args)
