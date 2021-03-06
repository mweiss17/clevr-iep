#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os
import time
import argparse
import json
import random
import shutil
from tqdm import tqdm
import torch
torch.backends.cudnn.enabled = True
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import h5py

import iep.utils as utils
import iep.preprocess
from iep.data import ClevrDataset, ClevrDataLoader
from iep.models import ModuleNetSanity, Seq2Seq, LstmModel, CnnLstmModel, CnnLstmSaModel


parser = argparse.ArgumentParser()

# Input data
parser.add_argument('--train_question_h5', default='data/train_questions.h5')
parser.add_argument('--train_features_h5')
parser.add_argument('--train_images_h5', default='data/train_images.h5')
parser.add_argument('--train_ocr_token_json')
parser.add_argument('--val_question_h5', default='data/val_questions.h5')
parser.add_argument('--val_features_h5')
parser.add_argument('--val_images_h5', default='data/val_images.h5')
parser.add_argument('--val_ocr_token_json')
parser.add_argument('--feature_dim', default='1024,14,14')
parser.add_argument('--vocab_json', default='data/vocab.json')
parser.add_argument('--multi_gpu', action='store_true')

parser.add_argument('--loader_num_workers', type=int, default=1)
parser.add_argument('--use_local_copies', default=0, type=int)
parser.add_argument('--cleanup_local_copies', default=1, type=int)

parser.add_argument('--family_split_file', default=None)
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--num_val_samples', default=10000, type=int)
parser.add_argument('--shuffle_train_data', default=1, type=int)

# What type of model to use and which parts to train
parser.add_argument('--model_type', default='PG',
        choices=['PG', 'EE', 'PG+EE', 'LSTM', 'CNN+LSTM', 'CNN+LSTM+SA', "PG+EE+GQNT"])
parser.add_argument('--train_program_generator', default=1, type=int)
parser.add_argument('--train_execution_engine', default=1, type=int)
parser.add_argument('--baseline_train_only_rnn', default=0, type=int)

# Start from an existing checkpoint
parser.add_argument('--program_generator_start_from', default=None)
parser.add_argument('--execution_engine_start_from', default=None)
parser.add_argument('--baseline_start_from', default=None)

# RNN options
parser.add_argument('--rnn_wordvec_dim', default=300, type=int)
parser.add_argument('--rnn_hidden_dim', default=256, type=int)
parser.add_argument('--rnn_num_layers', default=2, type=int)
parser.add_argument('--rnn_dropout', default=0, type=float)

# Module net options
parser.add_argument('--module_stem_num_layers', default=2, type=int)
parser.add_argument('--module_stem_batchnorm', default=0, type=int)
parser.add_argument('--module_dim', default=128, type=int)
parser.add_argument('--text_dim', default=1, type=int)
parser.add_argument('--module_residual', default=1, type=int)
parser.add_argument('--module_batchnorm', default=0, type=int)

# CNN options (for baselines)
parser.add_argument('--cnn_res_block_dim', default=128, type=int)
parser.add_argument('--cnn_num_res_blocks', default=0, type=int)
parser.add_argument('--cnn_proj_dim', default=512, type=int)
parser.add_argument('--cnn_pooling', default='maxpool2',
        choices=['none', 'maxpool2'])

# Stacked-Attention options
parser.add_argument('--stacked_attn_dim', default=512, type=int)
parser.add_argument('--num_stacked_attn', default=2, type=int)

# Classifier options
parser.add_argument('--classifier_proj_dim', default=512, type=int)
parser.add_argument('--classifier_downsample', default='maxpool2',
        choices=['maxpool2', 'maxpool4', 'none'])
parser.add_argument('--classifier_fc_dims', default='1024')
parser.add_argument('--classifier_batchnorm', default=0, type=int)
parser.add_argument('--classifier_dropout', default=0, type=float)

# Optimization options
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_iterations', default=100000, type=int)
parser.add_argument('--learning_rate', default=5e-4, type=float)
parser.add_argument('--reward_decay', default=0.9, type=float)

# Output options
parser.add_argument('--checkpoint_path', default='data/checkpoint.pt')
parser.add_argument('--randomize_checkpoint_path', type=int, default=0)
parser.add_argument('--record_loss_every', type=int, default=1)
parser.add_argument('--checkpoint_every', default=10000, type=int)


def main(args):
  if args.randomize_checkpoint_path == 1:
    name, ext = os.path.splitext(args.checkpoint_path)
    num = random.randint(1, 1000000)
    args.checkpoint_path = '%s_%06d%s' % (name, num, ext)

  vocab = utils.load_vocab(args.vocab_json)

  if args.use_local_copies == 1:
    shutil.copy(args.train_question_h5, '/tmp/train_questions.h5')
    shutil.copy(args.train_features_h5, '/tmp/train_features.h5')
    shutil.copy(args.val_question_h5, '/tmp/val_questions.h5')
    shutil.copy(args.val_features_h5, '/tmp/val_features.h5')
    args.train_question_h5 = '/tmp/train_questions.h5'
    args.train_features_h5 = '/tmp/train_features.h5'
    args.val_question_h5 = '/tmp/val_questions.h5'
    args.val_features_h5 = '/tmp/val_features.h5'

  question_families = None
  if args.family_split_file is not None:
    with open(args.family_split_file, 'r') as f:
      question_families = json.load(f)

  train_loader_kwargs = {
    'question_h5': args.train_question_h5,
    'feature_h5': args.train_features_h5,
    'vocab': vocab,
    'batch_size': args.batch_size,
    'shuffle': args.shuffle_train_data == 1,
    'question_families': question_families,
    'max_samples': args.num_train_samples,
    'num_workers': args.loader_num_workers,
    'image_h5': args.train_images_h5,
    'ocr_token_json': args.train_ocr_token_json
  }
  val_loader_kwargs = {
    'question_h5': args.val_question_h5,
    'feature_h5': args.val_features_h5,
    'vocab': vocab,
    'batch_size': args.batch_size,
    'question_families': question_families,
    'max_samples': args.num_val_samples,
    'num_workers': args.loader_num_workers,
    'image_h5': args.val_images_h5,
    'ocr_token_json': args.val_ocr_token_json
  }
  with ClevrDataLoader(**train_loader_kwargs) as train_loader, \
       ClevrDataLoader(**val_loader_kwargs) as val_loader:
    train_loop(args, train_loader, val_loader)
  import pdb; pdb.set_trace()
  if args.use_local_copies == 1 and args.cleanup_local_copies == 1:
    os.remove('/tmp/train_questions.h5')
    os.remove('/tmp/train_features.h5')
    os.remove('/tmp/val_questions.h5')
    os.remove('/tmp/val_features.h5')


def train_loop(args, train_loader, val_loader):
  vocab = utils.load_vocab(args.vocab_json)
  program_generator, pg_kwargs, pg_optimizer = None, None, None
  execution_engine, ee_kwargs, ee_optimizer = None, None, None
  baseline_model, baseline_kwargs, baseline_optimizer = None, None, None
  baseline_type = None

  pg_best_state, ee_best_state, baseline_best_state = None, None, None

  # Set up model
  if args.model_type == 'PG' or args.model_type == 'PG+EE' or args.model_type == 'PG+EE+GQNT':
    program_generator, pg_kwargs = get_program_generator(args)
    pg_optimizer = torch.optim.Adam(program_generator.parameters(),
                                    lr=args.learning_rate)
    print('Here is the program generator:')
    print(program_generator)
  if args.model_type == 'EE' or args.model_type == 'PG+EE'or args.model_type == 'PG+EE+GQNT':
    execution_engine, ee_kwargs = get_execution_engine(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.multi_gpu:
      execution_engine = torch.nn.DataParallel(execution_engine)
    elif device is not "cpu":
      execution_engine = execution_engine.cuda()

    ee_optimizer = torch.optim.Adam(execution_engine.parameters(),
                                    lr=args.learning_rate)
    print('Here is the execution engine:')
    print(execution_engine)
  if args.model_type in ['LSTM', 'CNN+LSTM', 'CNN+LSTM+SA']:
    baseline_model, baseline_kwargs = get_baseline_model(args)
    params = baseline_model.parameters()
    if args.baseline_train_only_rnn == 1:
      params = baseline_model.rnn.parameters()
    baseline_optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    print('Here is the baseline model')
    print(baseline_model)
    baseline_type = args.model_type
  if args.model_type == 'PG+EE+GQNT':
    kwargs = {
      'n_channels': 3,
      'v_dim': 224
    }
    model = MMModel(**kwargs)
    params = model.parameters()
    baseline_optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    print('Here is the GQNT model')
    print(model)

  loss_fn_1 = torch.nn.BCEWithLogitsLoss().cuda()
  loss_fn_2 = torch.nn.MSELoss()

  stats = {
    'train_losses': [], 'train_rewards': [], 'train_losses_ts': [],
    'train_accs': [], 'val_accs': [], 'val_accs_ts': [],
    'best_val_acc': -1, 'model_t': 0,
  }
  t, epoch, reward_moving_average = 0, 0, 0

  set_mode('train', [program_generator, execution_engine, baseline_model])

  print('train_loader has %d samples' % len(train_loader.dataset))
  print('val_loader has %d samples' % len(val_loader.dataset))

  while t < args.num_iterations:
    epoch += 1
    print('Starting epoch %d' % epoch)
    start = time.time()
    for batch in train_loader:
      # print("data loader: " + str(time.time() - start))
      start_batch = time.time()

      t += 1
      questions, images, feats, answers, programs, _, ocr_tokens = batch
      # print("mean answer value" + str((answers.sum() / float(len(answers))).item()))
      questions_var = Variable(questions.to(device))
      if programs[0] is not None:
        programs_var = Variable(programs.to(device))

      reward = None
      if args.model_type == 'PG':
        # Train program generator with ground-truth programs
        pg_optimizer.zero_grad()
        loss = program_generator(questions_var, programs_var)
        loss.backward()
        pg_optimizer.step()
      elif args.model_type == 'EE':
        # Train execution engine with ground-truth programs
        start = time.time()

        ee_optimizer.zero_grad()
        if images[0] is not None:
          images_var = Variable(images.to(device))
        else:
          feats_var = Variable(feats.to(device))
        answers_var = Variable(answers.to(device))
        text_embs = process_tokens(ocr_tokens).to(device)
        # text_embs = torch.cat([chars, locs], dim=1).reshape(-1).to(device)


        # print("OCR loading / processing + put stuff on cuda: " + str(time.time() - start))
        start = time.time()
        scores = execution_engine(feats_var, programs_var, text_embs)

        # print("Total Resnet + BiLSTM: " + str(time.time() - start))
        start = time.time()
        char_loss = loss_fn_1(scores, text_embs.reshape((-1,84)))
        loc_loss = loss_fn_2(text_embs[:, :, 26:], scores.reshape(-1, 3, 28)[:, :, 26:])
        loss = torch.sum(char_loss, loc_loss)

        loss.backward()
        ee_optimizer.step()
        # print("Optimization Step: " + str(time.time() - start))

      elif args.model_type in ['LSTM', 'CNN+LSTM', 'CNN+LSTM+SA']:
        baseline_optimizer.zero_grad()
        baseline_model.zero_grad()
        scores = baseline_model(questions_var, feats_var)
        loss = loss_fn(scores, answers_var)
        loss.backward()
        baseline_optimizer.step()
      elif args.model_type == 'PG+EE':
        programs_pred = program_generator.reinforce_sample(questions_var)
        scores = execution_engine(feats_var, programs_pred)

        loss = loss_fn(scores, answers_var)
        _, preds = scores.data.cpu().max(1)
        raw_reward = (preds == answers).float()
        reward_moving_average *= args.reward_decay
        reward_moving_average += (1.0 - args.reward_decay) * raw_reward.mean()
        centered_reward = raw_reward - reward_moving_average

        if args.train_execution_engine == 1:
          ee_optimizer.zero_grad()
          loss.backward()
          ee_optimizer.step()

        if args.train_program_generator == 1:
          pg_optimizer.zero_grad()
          program_generator.reinforce_backward(centered_reward.cuda())
          pg_optimizer.step()
      elif args.model_type == 'PG+EE+GQNT':
        programs_pred = program_generator.reinforce_sample(questions_var)
        baseline_optimizer.zero_grad()
        model.zero_grad()
        chars, locs = process_tokens(ocr_tokens)
        text_embs = torch.cat([chars, locs], dim=1).to(device)
        scores = execution_engine(feats_var, programs_pred, text_embs)
        loss = loss_fn(scores, answers_var)
        loss.backward()
        ee_optimizer.step()
      # print("total batch time (without data loading): " + str(time.time() - start_batch))

      if t % args.record_loss_every == 0:
        print(t, loss.data.item())
        stats['train_losses'].append(loss.data.item())
        stats['train_losses_ts'].append(t)
        if reward is not None:
          stats['train_rewards'].append(reward)

      if t % args.checkpoint_every == 0:
        print('Checking training accuracy ... ')
        train_acc, train_loc_err = check_accuracy(args, program_generator, execution_engine,
                                   baseline_model, train_loader)
        print('train accuracy is', train_acc)
        print('train loc error is', train_loc_err)
        print('Checking validation accuracy ...')
        val_acc, val_loc_err = check_accuracy(args, program_generator, execution_engine,
                                 baseline_model, val_loader)
        print('val accuracy is ', val_acc)
        print('val loc error is', val_loc_err)

        stats['train_accs'].append(train_acc)
        stats['val_accs'].append(val_acc)
        stats['val_accs_ts'].append(t)

        if val_acc > stats['best_val_acc']:
          stats['best_val_acc'] = val_acc
          stats['model_t'] = t
          best_pg_state = get_state(program_generator)
          best_ee_state = get_state(execution_engine)
          best_baseline_state = get_state(baseline_model)

        checkpoint = {
          'args': args.__dict__,
          'program_generator_kwargs': pg_kwargs,
          'program_generator_state': best_pg_state,
          'execution_engine_kwargs': ee_kwargs,
          'execution_engine_state': best_ee_state,
          'baseline_kwargs': baseline_kwargs,
          'baseline_state': best_baseline_state,
          'baseline_type': baseline_type,
          'vocab': vocab
        }
        for k, v in stats.items():
          checkpoint[k] = v
        print('Saving checkpoint to %s' % args.checkpoint_path)
        torch.save(checkpoint, args.checkpoint_path)
        del checkpoint['program_generator_state']
        del checkpoint['execution_engine_state']
        del checkpoint['baseline_state']
        with open(args.checkpoint_path + '.json', 'w') as f:
          json.dump(checkpoint, f)

      if t == args.num_iterations:
        break

def process_tokens(ocr_tokens):
  depth = 26
  ones = torch.sparse.torch.eye(depth)
  all_text_embs = torch.FloatTensor([])
  all_chars = torch.FloatTensor([])
  all_locs = torch.FloatTensor([])

  # order the tokens by raster order
  for scene in ocr_tokens:
    chars_d = {}
    for token in scene:
      char = ord(token['body']) - ord('a')
      chars_d["." + str(token['pixel_coords'][1]).split(".")[-1] + "." + str(token['pixel_coords'][0]).split(".")[-1]] = char

    scene_chars = []
    scene_locs = []
    for key in sorted(chars_d):
      scene_chars.append(chars_d[key])
      scene_locs.append((float("." + key.split(".")[2]), float("." + key.split(".")[1])))

    scene_chars = ones.index_select(0, torch.LongTensor(scene_chars))
    scene_locs_vec = torch.FloatTensor(scene_locs)#.view(1, -1, 2)
    scene_text_emb = torch.cat([scene_chars, scene_locs_vec], dim=1)

    all_text_embs = torch.cat([all_text_embs, scene_text_emb.view(1, -1, 28)], dim=0)

  return all_text_embs

def parse_int_list(s):
  return tuple(int(n) for n in s.split(','))


def get_state(m):
  if m is None:
    return None
  state = {}
  for k, v in m.state_dict().items():
    state[k] = v.clone()
  return state


def get_program_generator(args):
  vocab = utils.load_vocab(args.vocab_json)
  if args.program_generator_start_from is not None:
    pg, kwargs = utils.load_program_generator(args.program_generator_start_from)
    cur_vocab_size = pg.encoder_embed.weight.size(0)
    if cur_vocab_size != len(vocab['question_token_to_idx']):
      print('Expanding vocabulary of program generator')
      pg.expand_encoder_vocab(vocab['question_token_to_idx'])
      kwargs['encoder_vocab_size'] = len(vocab['question_token_to_idx'])
  else:
    kwargs = {
      'encoder_vocab_size': len(vocab['question_token_to_idx']),
      'decoder_vocab_size': len(vocab['program_token_to_idx']),
      'wordvec_dim': args.rnn_wordvec_dim,
      'hidden_dim': args.rnn_hidden_dim,
      'rnn_num_layers': args.rnn_num_layers,
      'rnn_dropout': args.rnn_dropout,
    }
    pg = Seq2Seq(**kwargs)
  pg.cuda()
  pg.train()
  return pg, kwargs


def get_execution_engine(args):
  vocab = utils.load_vocab(args.vocab_json)
  if args.execution_engine_start_from is not None:
    ee, kwargs = utils.load_execution_engine(args.execution_engine_start_from)
    # TODO: Adjust vocab?
  else:
    kwargs = {
      'vocab': vocab,
      'feature_dim': parse_int_list(args.feature_dim),
      'stem_batchnorm': args.module_stem_batchnorm == 1,
      'stem_num_layers': args.module_stem_num_layers,
      'module_dim': args.module_dim,
      'text_dim': args.text_dim,
      'module_residual': args.module_residual == 1,
      'module_batchnorm': args.module_batchnorm == 1,
      'classifier_proj_dim': args.classifier_proj_dim,
      'classifier_downsample': args.classifier_downsample,
      'classifier_fc_layers': parse_int_list(args.classifier_fc_dims),
      'classifier_batchnorm': args.classifier_batchnorm == 1,
      'classifier_dropout': args.classifier_dropout,
    }
    ee = ModuleNetSanity(**kwargs)
  ee.cuda()
  ee.train()
  return ee, kwargs


def get_baseline_model(args):
  vocab = utils.load_vocab(args.vocab_json)
  if args.baseline_start_from is not None:
    model, kwargs = utils.load_baseline(args.baseline_start_from)
  elif args.model_type == 'LSTM':
    kwargs = {
      'vocab': vocab,
      'rnn_wordvec_dim': args.rnn_wordvec_dim,
      'rnn_dim': args.rnn_hidden_dim,
      'rnn_num_layers': args.rnn_num_layers,
      'rnn_dropout': args.rnn_dropout,
      'fc_dims': parse_int_list(args.classifier_fc_dims),
      'fc_use_batchnorm': args.classifier_batchnorm == 1,
      'fc_dropout': args.classifier_dropout,
    }
    model = LstmModel(**kwargs)
  elif args.model_type == 'CNN+LSTM':
    kwargs = {
      'vocab': vocab,
      'rnn_wordvec_dim': args.rnn_wordvec_dim,
      'rnn_dim': args.rnn_hidden_dim,
      'rnn_num_layers': args.rnn_num_layers,
      'rnn_dropout': args.rnn_dropout,
      'cnn_feat_dim': parse_int_list(args.feature_dim),
      'cnn_num_res_blocks': args.cnn_num_res_blocks,
      'cnn_res_block_dim': args.cnn_res_block_dim,
      'cnn_proj_dim': args.cnn_proj_dim,
      'cnn_pooling': args.cnn_pooling,
      'fc_dims': parse_int_list(args.classifier_fc_dims),
      'fc_use_batchnorm': args.classifier_batchnorm == 1,
      'fc_dropout': args.classifier_dropout,
    }
    model = CnnLstmModel(**kwargs)
  elif args.model_type == 'CNN+LSTM+SA':
    kwargs = {
      'vocab': vocab,
      'rnn_wordvec_dim': args.rnn_wordvec_dim,
      'rnn_dim': args.rnn_hidden_dim,
      'rnn_num_layers': args.rnn_num_layers,
      'rnn_dropout': args.rnn_dropout,
      'cnn_feat_dim': parse_int_list(args.feature_dim),
      'stacked_attn_dim': args.stacked_attn_dim,
      'num_stacked_attn': args.num_stacked_attn,
      'fc_dims': parse_int_list(args.classifier_fc_dims),
      'fc_use_batchnorm': args.classifier_batchnorm == 1,
      'fc_dropout': args.classifier_dropout,
    }
    model = CnnLstmSaModel(**kwargs)
  if model.rnn.token_to_idx != vocab['question_token_to_idx']:
    # Make sure new vocab is superset of old
    for k, v in model.rnn.token_to_idx.items():
      assert k in vocab['question_token_to_idx']
      assert vocab['question_token_to_idx'][k] == v
    for token, idx in vocab['question_token_to_idx'].items():
      model.rnn.token_to_idx[token] = idx
    kwargs['vocab'] = vocab
    model.rnn.expand_vocab(vocab['question_token_to_idx'])
  model.cuda()
  model.train()
  return model, kwargs


def set_mode(mode, models):
  assert mode in ['train', 'eval']
  for m in models:
    if m is None: continue
    if mode == 'train': m.train()
    if mode == 'eval': m.eval()


def check_accuracy(args, program_generator, execution_engine, baseline_model, loader):
  set_mode('eval', [program_generator, execution_engine, baseline_model])
  num_correct, num_samples, num_locs = 0, 0, 0.
  loc_distance = 0
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  for batch in tqdm(loader):
    questions, images, feats, answers, programs, _, ocr_tokens = batch
    with torch.no_grad():
      questions_var = Variable(questions.cuda())
      if feats[0] is not None:
        feats_var = Variable(feats.cuda())
      answers_var = Variable(answers.cuda())
      if programs[0] is not None:
        programs_var = Variable(programs.cuda())
    scores = None # Use this for everything but PG
    if args.model_type == 'PG':
      vocab = utils.load_vocab(args.vocab_json)
      for i in range(questions.size(0)):
        with torch.no_grad():
          program_pred = program_generator.sample(Variable(questions[i:i+1].cuda()))
        program_pred_str = iep.preprocess.decode(program_pred, vocab['program_idx_to_token'])
        program_str = iep.preprocess.decode(programs[i].tolist(), vocab['program_idx_to_token'])
        if program_pred_str == program_str:
          num_correct += 1
        num_samples += 1
    elif args.model_type == 'EE':
      text_embs = process_tokens(ocr_tokens).to(device)
      scores = execution_engine(feats_var, programs_var, text_embs)

    elif args.model_type == 'PG+EE':
      programs_pred = program_generator.reinforce_sample(
                          questions_var, argmax=True)
      scores = execution_engine(feats_var, programs_pred)
    elif args.model_type in ['LSTM', 'CNN+LSTM', 'CNN+LSTM+SA']:
      scores = baseline_model(questions_var, feats_var)
    elif args.model_type == 'PG+EE+GQNT':
      programs_pred = program_generator.reinforce_sample(questions_var)
      text_embs = process_tokens(ocr_tokens).to(device)
      scores = execution_engine(feats_var, programs_pred, text_embs)

    if scores is not None:
      pred_chars = scores.reshape(-1, 3, 28)[:, :, :26].argmax(dim=2).view(-1).data.cpu()
      gt_chars = text_embs[:,:,:26].argmax(dim=2).view(-1).data.cpu()
      locs = scores.reshape(-1, 3, 28)[:, :, 26:].data.cpu()
      loc_distance += torch.dist(text_embs[:, :, 26:].data.cpu(), locs)

      num_locs += locs.size(0)
      num_correct += (pred_chars == gt_chars).sum()
      num_samples += pred_chars.size(0)

    if num_samples >= args.num_val_samples:
      break

  set_mode('train', [program_generator, execution_engine, baseline_model])
  acc = float(num_correct) / num_samples
  loc_err = float(loc_distance) / num_locs
  return acc, loc_err


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
