#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import argparse

import json
import os

import h5py
from tqdm import tqdm
import numpy as np

import iep.programs
from iep.preprocess import tokenize, encode, build_vocab


"""
Preprocessing script for CLEVR question files.
"""


parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='prefix',
                    choices=['chain', 'prefix', 'postfix'])
parser.add_argument('--input_questions_json', required=True)
parser.add_argument('--input_scenes_json', required=True)
parser.add_argument('--input_vocab_json', default='')
parser.add_argument('--expand_vocab', default=0, type=int)
parser.add_argument('--unk_threshold', default=1, type=int)
parser.add_argument('--encode_unk', default=0, type=int)
parser.add_argument('--multi_dir', action="store_true")
parser.add_argument('--num_views', default=1, type=int)
parser.add_argument('--binary_qs_only', action="store_true")

parser.add_argument('--output_h5_file', required=True)
parser.add_argument('--output_vocab_json', default='')


def program_to_str(program, mode):
  if mode == 'chain':
    if not iep.programs.is_chain(program):
      return None
    return iep.programs.list_to_str(program)
  elif mode == 'prefix':
    program_prefix = iep.programs.list_to_prefix(program)
    return iep.programs.list_to_str(program_prefix)
  elif mode == 'postfix':
    program_postfix = iep.programs.list_to_postfix(program)
    return iep.programs.list_to_str(program_postfix)
  return None


def main(args):
  if (args.input_vocab_json == '') and (args.output_vocab_json == ''):
    print('Must give one of --input_vocab_json or --output_vocab_json')
    return
  if "train" in args.output_h5_file and args.multi_dir:
      subdirs = [x for x in range(25)]
  elif "val" in args.output_h5_file and args.multi_dir:
      subdirs = [25, 26]
  elif args.multi_dir:
      subdirs = [27, 28, 29]
  else:
    subdirs = []

  questions = []
  scenes = []
  for subdir in subdirs:
    question_path = os.path.join(args.input_questions_json, str(subdir), "questions.json")
    scene_path = os.path.join(args.input_scenes_json, str(subdir), "scenes.json")
    ss = json.load(open(scene_path, "r"))['scenes']
    for s in ss:
      s['cc']['subdir'] = subdir
    scenes.extend(ss)
    qs = json.load(open(question_path, "r"))['questions']
    for q in qs:
        q['subdir'] = subdir
    questions.extend(qs)
  if not questions:
    questions = json.load(open(args.input_questions_json, "r"))['questions']
  if not scenes:
    scenes = json.load(open(args.input_scenes_json, "r"))['scenes']
  if args.binary_qs_only:
    filtered_questions = []
    for q in tqdm(questions):
      if q['answer'] in [True, False] and q['question'] != "?":
        filtered_questions.append(q)
    questions = filtered_questions
  # Either create the vocab or load it from disk
  if args.input_vocab_json == '' or args.expand_vocab == 1:
    print('Building vocab')
    if 'answer' in questions[0]:
      answer_token_to_idx = build_vocab(
        (str(q['answer']) for q in questions),
      answers_only=True)
    question_token_to_idx = build_vocab(
      (q['question'] for q in questions),
      min_token_count=args.unk_threshold,
      punct_to_keep=[';', ','], punct_to_remove=['?', '.']
    )
    all_program_strs = []
    for q in questions:
      if 'program' not in q: continue
      program_str = program_to_str(q['program'], args.mode)
      if program_str is not None:
        all_program_strs.append(program_str)
    program_token_to_idx = build_vocab(all_program_strs)

    all_scene_text = []
    for scene in scenes:
      for view_name, view_struct in scene.items():
        for object in view_struct['objects']:
          all_scene_text.append(object['text']['body'])
    ocr_to_idx = build_vocab(all_scene_text)

    vocab = {
      'ocr_to_idx': ocr_to_idx,
      'question_token_to_idx': question_token_to_idx,
      'program_token_to_idx': program_token_to_idx,
      'answer_token_to_idx': answer_token_to_idx,
    }
  if args.input_vocab_json != '':
    print('Loading vocab')
    if args.expand_vocab == 1:
      new_vocab = vocab
    with open(args.input_vocab_json, 'r') as f:
      vocab = json.load(f)
    if args.expand_vocab == 1:
      num_new_words = 0
      for word in new_vocab['question_token_to_idx']:
        if word not in vocab['question_token_to_idx']:
          print('Found new word %s' % word)
          idx = len(vocab['question_token_to_idx'])
          vocab['question_token_to_idx'][word] = idx
          num_new_words += 1
      print('Found %d new words' % num_new_words)

  vocab_out_path = args.output_vocab_json.split(".")[0] + ".txt"
  if vocab_out_path is not ".txt":
    with open(vocab_out_path, "w") as out_file:
      for word in vocab['ocr_to_idx'].keys():
        out_file.write(word + "\n")

  if args.output_vocab_json != '':
    with open(args.output_vocab_json, 'w') as f:
      json.dump(vocab, f)

  # Encode all questions and programs
  print('Encoding data')
  questions_encoded = []
  programs_encoded = []
  question_families = []
  orig_idxs = []
  image_idxs = []
  answers = []
  baseline = questions[0]['image_index']
  for orig_idx, q in enumerate(questions):
    question = q['question']
    # We need to ask the same question about each view of the scene, and there are 20 views of each scene
    if q.get("subdir"):
      offset = q['image_index'] - baseline
      # num_images_per_subdir = len(os.listdir(os.path.join(args.input_scenes_json, str(subdir), "images")))
      # image_name = questions[0]['image']
      # count = 0
      # for i in range(200):
      #   image_name_2 = questions[i]['image']
      #   if image_name != image_name_2:
      #     break
      #   count += 1
      # num_questions_per_image = count
      # import pdb; pdb.set_trace()
      # offset = num_images_per_subdir * q['subdir'] + q['image_index'] * num_questions_per_image
    else:
      offset = q['image_index']

    for view in range(args.num_views):

      orig_idxs.append(orig_idx)
      image_idxs.append(offset + view)
      if 'question_family_index' in q:
        question_families.append(q['question_family_index'])
      question_tokens = tokenize(question,
                          punct_to_keep=[';', ','],
                          punct_to_remove=['?', '.'])
      question_encoded = encode(question_tokens,
                           vocab['question_token_to_idx'],
                           allow_unk=args.encode_unk == 1)
      questions_encoded.append(question_encoded)

      if 'program' in q:
        program = q['program']
        program_str = program_to_str(program, args.mode)
        program_tokens = tokenize(program_str)
        program_encoded = encode(program_tokens, vocab['program_token_to_idx'])
        programs_encoded.append(program_encoded)

      if 'answer' in q:
        try:
          answers.append(vocab['answer_token_to_idx'][str(q['answer'])])
        except Exception as e:
          print(e)
  # Pad encoded questions and programs
  max_question_length = max(len(x) for x in questions_encoded)
  for qe in questions_encoded:
    while len(qe) < max_question_length:
      qe.append(vocab['question_token_to_idx']['<NULL>'])

  if len(programs_encoded) > 0:
    max_program_length = max(len(x) for x in programs_encoded)
    for pe in programs_encoded:
      while len(pe) < max_program_length:
        pe.append(vocab['program_token_to_idx']['<NULL>'])

  # Create h5 file
  print('Writing output')
  questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
  programs_encoded = np.asarray(programs_encoded, dtype=np.int32)
  print(questions_encoded.shape)
  print(programs_encoded.shape)
  with h5py.File(args.output_h5_file, 'w') as f:
    f.create_dataset('questions', data=questions_encoded)
    f.create_dataset('image_idxs', data=np.asarray(image_idxs))
    f.create_dataset('orig_idxs', data=np.asarray(orig_idxs))

    if len(programs_encoded) > 0:
      f.create_dataset('programs', data=programs_encoded)
    if len(question_families) > 0:
      f.create_dataset('question_families', data=np.asarray(question_families))
    if len(answers) > 0:
      f.create_dataset('answers', data=np.asarray(answers))


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
