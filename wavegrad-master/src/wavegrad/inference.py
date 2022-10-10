# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import os
import torch
import torchaudio

from argparse import ArgumentParser

from wavegrad.params import AttrDict, params as base_params
from wavegrad.model import WaveGrad


models = {}

def predict(spectrogram, model_dir=None, params=None, device=torch.device('cuda')):
  # Lazy load model.
  if not model_dir in models:
    if os.path.exists(f'{model_dir}/weights.pt'):
      checkpoint = torch.load(f'{model_dir}/weights.pt')
    else:
      checkpoint = torch.load(model_dir)
    model = WaveGrad(AttrDict(base_params)).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    models[model_dir] = model

  model = models[model_dir]
  model.params.override(params)
  with torch.no_grad():
    beta = np.array(model.params.noise_schedule)
    alpha = 1 - beta
    alpha_cum = np.cumprod(alpha)

    # Expand rank 2 tensors by adding a batch dimension.
    if len(spectrogram.shape) == 2:
      spectrogram = spectrogram.unsqueeze(0)
    spectrogram = spectrogram.to(device)

    audio = torch.randn(spectrogram.shape[0], model.params.hop_samples * spectrogram.shape[-1], device=device)
    noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)

    for n in range(len(alpha) - 1, -1, -1):
      c1 = 1 / alpha[n]**0.5
      c2 = (1 - alpha[n]) / (1 - alpha_cum[n])**0.5
      audio = c1 * (audio - c2 * model(audio, spectrogram, noise_scale[n]).squeeze(1))
      if n > 0:
        noise = torch.randn_like(audio)
        sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
        audio += sigma * noise
      audio = torch.clamp(audio, -1.0, 1.0)
  return audio, model.params.sample_rate


def main(args):
  spectrogram = torch.from_numpy(np.load(args.spectrogram_path))
  params = {}
  if args.noise_schedule:
    params['noise_schedule'] = torch.from_numpy(np.load(args.noise_schedule))
  audio, sr = predict(spectrogram, model_dir=args.model_dir, params=params)
  torchaudio.save(args.output, audio.cpu(), sample_rate=sr)


if __name__ == '__main__':
  parser = ArgumentParser(description='runs inference on a spectrogram file generated by wavegrad.preprocess')
  parser.add_argument('model_dir',
      help='directory containing a trained model (or full path to weights.pt file)')
  parser.add_argument('spectrogram_path',
      help='path to a spectrogram file generated by wavegrad.preprocess')
  parser.add_argument('--noise-schedule', '-n', default=None,
      help='path to a custom noise schedule file generated by wavegrad.noise_schedule')
  parser.add_argument('--output', '-o', default='output.wav',
      help='output file name')
  main(parser.parse_args())
