#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: hawcat
@time: 2024/1/25 10:58 
@file: inference_cli.py
@project: GPT-SoVITS
@describe: TODO
"""
import argparse
import datetime
import json
import os
import subprocess
import traceback
from typing import Tuple, List, Dict, Union, Any

import yaml
import s1_train
import s2_train
import tts_gen
import utils

import spacy
from faster_whisper import WhisperModel

from config import python_exec, is_half, exp_root

model_path = "D:/O3BackEND/repository/models/faster-whisper-large-v3/"
ffmpeg_path = "ffmpeg.exe"


def open1abc(inp_text, inp_wav_dir, exp_name, gpu_numbers1a="0-0", gpu_numbers1Ba="0-0", gpu_numbers1c="0-0",
             bert_pretrained_dir="pretrained_models/chinese-roberta-wwm-ext-large",
             ssl_pretrained_dir="pretrained_models/chinese-hubert-base",
             pretrained_s2G_path="pretrained_models/s2G488k.pth"):
    ps1abc = None
    if not ps1abc:
        opt_dir = "%s/%s" % (os.path.join(work_dir, exp_root), exp_name)
        try:
            #############################1a
            path_text = "%s/2-name2text.txt" % opt_dir
            if (os.path.exists(path_text) == False or (os.path.exists(path_text) == True and len(
                    open(path_text, "r", encoding="utf8").read().strip("\n").split("\n")) < 2)):
                config = {
                    "inp_text": inp_text,
                    "inp_wav_dir": inp_wav_dir,
                    "exp_name": exp_name,
                    "opt_dir": opt_dir,
                    "bert_pretrained_dir": bert_pretrained_dir,
                    "is_half": str(is_half)
                }
                gpu_names = gpu_numbers1a.split("-")
                all_parts = len(gpu_names)
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                        }
                    )
                    os.environ.update(config)
                    # get_text.main()
                    subprocess.run(["python", "prepare_datasets/get_text.py"], shell=True)

                opt = []
                for i_part in range(all_parts):  # txt_path="%s/2-name2text-%s.txt"%(opt_dir,i_part)
                    txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
                    with open(txt_path, "r", encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(txt_path)
                with open(path_text, "w", encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")

            #############################1b
            config = {
                "inp_text": inp_text,
                "inp_wav_dir": inp_wav_dir,
                "exp_name": exp_name,
                "opt_dir": opt_dir,
                "cnhubert_base_dir": ssl_pretrained_dir,
            }
            gpu_names = gpu_numbers1Ba.split("-")
            all_parts = len(gpu_names)
            for i_part in range(all_parts):
                config.update(
                    {
                        "i_part": str(i_part),
                        "all_parts": str(all_parts),
                        "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                    }
                )
                os.environ.update(config)
                # get_hubert_wav32k.main()
                subprocess.run(["python", "prepare_datasets/get_hubert_wav32k.py"], shell=True)
            #############################1c
            path_semantic = "%s/6-name2semantic.tsv" % opt_dir
            if (os.path.exists(path_semantic) == False or (
                    os.path.exists(path_semantic) == True and os.path.getsize(path_semantic) < 31)):
                config = {
                    "inp_text": inp_text,
                    "exp_name": exp_name,
                    "opt_dir": opt_dir,
                    "pretrained_s2G": pretrained_s2G_path,
                    "s2config_path": "configs/s2.json",
                }
                gpu_names = gpu_numbers1c.split("-")
                all_parts = len(gpu_names)
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                        }
                    )
                    os.environ.update(config)
                    # get_semantic.main()
                    subprocess.run(["python", "prepare_datasets/get_semantic.py"], shell=True)

                opt = ["item_name\tsemantic_audio"]
                for i_part in range(all_parts):
                    semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
                    with open(semantic_path, "r", encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(semantic_path)
                with open(path_semantic, "w", encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")

            print("======================================预处理执行完成=====================================")

        except:
            traceback.print_exc()
            raise "一键三连中途报错"
            # yield "一键三连中途报错", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}
    else:
        raise "已有正在进行的一键三连任务，需先终止才能开启下一次任务"


def sovits_train(exp_name, batch_size=12, total_epoch=12, text_low_lr_rate=0.4, if_save_latest=True,
                 if_save_every_weights=True,
                 save_every_epoch=12, gpu_numbers1Ba="0",
                 pretrained_s2G="pretrained_models/s2G488k.pth",
                 pretrained_s2D="pretrained_models/s2D488k.pth"):
    p_train_SoVITS = None  # TODO
    if p_train_SoVITS is None:
        with open("configs/s2.json") as f:
            data = f.read()
            data = json.loads(data)
        s2_dir = "%s/%s" % (os.path.join(work_dir, exp_root), exp_name)
        os.makedirs("%s/logs_s2" % s2_dir, exist_ok=True)
        data["train"]["batch_size"] = batch_size
        data["train"]["epochs"] = total_epoch
        data["train"]["text_low_lr_rate"] = text_low_lr_rate
        data["train"]["pretrained_s2G"] = pretrained_s2G
        data["train"]["pretrained_s2D"] = pretrained_s2D
        data["train"]["if_save_latest"] = if_save_latest
        data["train"]["if_save_every_weights"] = if_save_every_weights
        data["train"]["save_every_epoch"] = save_every_epoch
        data["train"]["gpu_numbers"] = gpu_numbers1Ba
        data["data"]["exp_dir"] = data["s2_ckpt_dir"] = s2_dir
        data["save_weight_dir"] = SoVITS_weight_root
        data["name"] = exp_name
        tmp_config_path = "TEMP/tmp_s2.json"
        with open(tmp_config_path, "w") as f:
            f.write(json.dumps(data))

        s2_train.main(tmp_config_path)

        # TODO
        result_path = None
        file_list = os.listdir(SoVITS_weight_root)
        for filename in file_list:
            if filename.startswith(f"{exp_name}_e12"):
                result_path = os.path.join(SoVITS_weight_root, filename)

        if result_path:
            return result_path
        else:
            raise "SoVITS训练失败"


def gpt_train(exp_name, batch_size=12, total_epoch=15, if_save_latest=True, if_save_every_weights=True,
              save_every_epoch=15, gpu_numbers="0",
              pretrained_s1="pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"):
    p_train_GPT = None  # TODO
    if p_train_GPT is None:
        with open("configs/s1longer.yaml") as f:
            data = f.read()
            data = yaml.load(data, Loader=yaml.FullLoader)
        s1_dir = "%s/%s" % (os.path.join(work_dir, exp_root), exp_name)
        os.makedirs("%s/logs_s1" % s1_dir, exist_ok=True)
        data["train"]["batch_size"] = batch_size
        data["train"]["epochs"] = total_epoch
        data["pretrained_s1"] = pretrained_s1
        data["train"]["save_every_n_epoch"] = save_every_epoch
        data["train"]["if_save_every_weights"] = if_save_every_weights
        data["train"]["if_save_latest"] = if_save_latest
        data["train"]["half_weights_save_dir"] = GPT_weight_root
        data["train"]["exp_name"] = exp_name
        data["train_semantic_path"] = "%s/6-name2semantic.tsv" % s1_dir
        data["train_phoneme_path"] = "%s/2-name2text.txt" % s1_dir
        data["output_dir"] = "%s/logs_s1" % s1_dir

        os.environ["_CUDA_VISIBLE_DEVICES"] = gpu_numbers.replace("-", ",")
        os.environ["hz"] = "25hz"
        tmp_config_path = "TEMP/tmp_s1.yaml"
        with open(tmp_config_path, "w") as f:
            f.write(yaml.dump(data, default_flow_style=False))

        s1_train.main(tmp_config_path)

        # TODO
        result_path = None
        file_list = os.listdir(GPT_weight_root)
        for filename in file_list:
            if filename.startswith(f"{exp_name}-e15"):
                result_path = os.path.join(GPT_weight_root, filename)

        if result_path:
            return result_path
        else:
            raise "GPT训练失败"


def get_weights_names():
    SoVITS_names = [pretrained_sovits_name]
    for name in os.listdir(SoVITS_weight_root):
        if name.endswith(".pth"):
            SoVITS_names.append("%s/%s" % (SoVITS_weight_root, name))
    GPT_names = [pretrained_gpt_name]
    for name in os.listdir(GPT_weight_root):
        if name.endswith(".ckpt"):
            GPT_names.append("%s/%s" % (GPT_weight_root, name))
    return SoVITS_names, GPT_names


def strfdelta(tdelta, fmt):
    d = {"days": tdelta.days}
    d["hours"], rem = divmod(tdelta.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)
    # d["milliseconds"] = random.uniform(1,999)
    return fmt.format(**d)


def format_seconds(seconds):
    # 将秒数转换为 timedelta 对象
    td = datetime.timedelta(seconds=seconds)

    # 使用 strfdelta 函数格式化时间
    formatted_time = strfdelta(td, "{hours:02d}:{minutes:02d}:{seconds:02d}")

    return formatted_time


def audio_recognition(input_wav_path: str) -> tuple[list[dict[str, Union[str, Any]]], str, dict[str, Union[str, Any]]]:
    model = WhisperModel(model_size_or_path=model_path, device="cuda", compute_type="float16")

    segments, info = model.transcribe(input_wav_path, beam_size=5)

    language = info.language.upper()

    asr_list = []
    os.makedirs(os.path.join(work_dir, "asr_opt"), exist_ok=True)
    output_slicer_path = os.path.join(work_dir, "asr_opt", "slicer_opt.list")

    for idx, segment in enumerate(segments):
        with open(output_slicer_path, mode="a+", encoding="utf-8") as f:
            split_path = os.path.join(work_dir, "slicer_opt", input_wav_path + f"_{idx}.wav")
            asr_list.append({"wav_path": split_path, "start_time": format_seconds(segment.start),
                             "end_time": format_seconds(segment.end)})
            f.write(
                f"{os.path.abspath(split_path)}|{os.path.basename(os.path.dirname(split_path))}|{language}|{segment.text}\n")

            # TODO
            if idx == 2:
                reference_content = {"wav_path": split_path, "text": segment.text, "language": language}

    return asr_list, output_slicer_path, reference_content


def split_wav(input_wav_path: str, asr_list: list):
    os.makedirs(os.path.join(work_dir, "slicer_opt"), exist_ok=True)
    for slice_audio in asr_list:
        try:
            subprocess.run([ffmpeg_path,
                            "-i", input_wav_path,
                            "-ss", slice_audio['start_time'],
                            "-to", slice_audio['end_time'],
                            "-c:a", "copy",
                            slice_audio['wav_path']])
            print(f"split {input_wav_path} to {slice_audio['wav_path']}")
        except Exception as e:
            print(e)


def split_sentences(text: str) -> list:
    """
    推理用
    Args:
        text:

    Returns:

    """
    spacy.prefer_gpu()
    nlp = spacy.load("zh_core_web_sm")
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="C:\\Users\\hawcat\\Desktop\\preparation")
    parser.add_argument("--input_wav_file", type=str, default="input.wav")
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--tts_prompt", type=str,
                        default="今天我很荣幸作为一个青藏高原的孩子能来到联合国讲我和动物朋友们的故事。")

    args = parser.parse_args()
    work_dir = args.work_dir
    input_wav_file = args.input_wav_file

    SoVITS_weight_root = os.path.join(work_dir, "SoVITS_weights")
    GPT_weight_root = os.path.join(work_dir, "GPT_weights")
    os.makedirs(SoVITS_weight_root, exist_ok=True)
    os.makedirs(GPT_weight_root, exist_ok=True)
    pretrained_sovits_name = "pretrained_models/s2G488k.pth"
    pretrained_gpt_name = "pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"

    ###############
    # Start ASR
    ###############
    recognition_info_list, slicer_path, reference = audio_recognition(input_wav_file)
    split_wav(input_wav_file, recognition_info_list)

    ################
    # Start training
    ################
    open1abc(slicer_path, os.path.join(work_dir, "slicer_opt"), args.exp_name)

    sovits_train(args.exp_name)
    gpt_train(args.exp_name)

    ################
    # Inference
    ################
    reference_wav = reference["wav_path"]
    reference_text = reference["text"]
    reference_lang = reference["language"]

    # TODO
    if utils.contains_chinese(args.tts_prompt):
        prompt_lang = "zh"
    else:
        prompt_lang = "en"

    sentences = split_sentences(args.tts_prompt)

    for idx, sentence in enumerate(sentences):
        tts_gen.get_tts_wav(reference_wav, sentence, prompt_lang, reference_text, reference_lang,
                            os.path.join(work_dir, f"output_{idx}.wav"))
