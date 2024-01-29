#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: hawcat
@time: 2024/1/29 12:45 
@file: inference_cli.py
@project: GPT-SoVITS
@describe: TODO
"""
import os
import wave

import spacy

import utils
import argparse
import tts_gen

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


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


def concatenate_wav_files(input_files: list, output_file: str):
    # 打开输出文件，创建一个新的wave文件
    with wave.open(output_file, 'wb') as output_wave:
        # 打开输入文件，获取参数
        for input_file in input_files:
            with wave.open(input_file, 'rb') as input_wave:
                # 如果是第一个文件，设置输出wave文件的参数
                if input_files.index(input_file) == 0:
                    output_wave.setparams(input_wave.getparams())

                # 读取输入文件的数据并写入输出文件
                data = input_wave.readframes(input_wave.getnframes())
                output_wave.writeframes(data)


def denosie(input, output):
    ans = pipeline(
        Tasks.acoustic_noise_suppression,
        model='damo/speech_frcrn_ans_cirm_16k')
    result = ans(
        input,
        output_path=output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, default="C:\\Users\\hawcat\\Desktop\\preparation")
    parser.add_argument('--tts_prompt', type=str,
                        default="今天我很荣幸作为一个青藏高原的孩子能来到联合国，讲我和动物朋友们的故事。我的村庄叫然日卡，小小的，但是格聂山和横断山脉却很大。")
    parser.add_argument("--output_file", type=str, default="C:\\Users\\hawcat\\Desktop\\preparation\\123.wav")
    args = parser.parse_args()

    if os.path.exists(os.path.join(args.work_dir, "train.list")):
        with open(os.path.join(args.work_dir, "train.list"), "r", encoding="utf-8") as f:
            model_info = f.readline()
            inference_info = model_info.split("|")
    else:
        raise FileNotFoundError("train.list not found")

    if utils.contains_chinese(args.tts_prompt):
        prompt_lang = "zh"
    else:
        prompt_lang = "en"

    sentences = split_sentences(args.tts_prompt)
    inference_files = []

    for idx, sentence in enumerate(sentences):
        file_path = os.path.join(args.work_dir, f"inference_{idx}.wav")
        inference_files.append(file_path)
        tts_gen.get_tts_wav(gpt_path=inference_info[0], sovits_path=inference_info[1], ref_wav_path=inference_info[2],
                            prompt_text=inference_info[3], prompt_language=inference_info[4], text=sentence,
                            text_language=prompt_lang,
                            output_file=file_path)

    temp_audio = os.path.join(args.work_dir, "temp.wav")
    concatenate_wav_files(inference_files, temp_audio)
    denosie(temp_audio, args.output_file)
