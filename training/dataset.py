# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
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
from typing import Any, List, Tuple, Union

import librosa
import numpy as np
import torch
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from torch.utils.data import Dataset

AudioLike = Union[
    str,                     # wav path, URL, base64
    np.ndarray,              # waveform (requires sr)
    Tuple[np.ndarray, int],  # (waveform, sr)
]

MaybeList = Union[Any, List[Any]]

# Emotion table row order. This order is baked into the saved checkpoint, so it
# must stay fixed once a model has been trained with it.
EMOTIONS = ["anger", "disgust", "fear", "happy", "sad", "surprise"]
EMOTION_TO_ID = {name: index for index, name in enumerate(EMOTIONS)}


class TTSDataset(Dataset):
    def __init__(self, data_list, processor, config:Qwen3TTSConfig, lag_num = -1, language_tag = False):
        self.data_list = data_list
        self.processor = processor
        self.lag_num = lag_num
        self.config = config
        # Off: the original "nothink" prefix with no language. On: the "think"
        # prefix with each row's language id, as generate() builds it when a
        # language other than Auto is given. The prefix is one slot longer, so
        # every later position (speaker slot included) moves right by one.
        self.language_tag = language_tag

    def _language_id(self, language: str) -> int:
        # Same lookup as generate(); Auto has no id there, so it cannot be tagged.
        table = self.config.talker_config.codec_language_id
        if language.lower() not in table:
            raise ValueError(
                f"language_tag needs a tagged language, got {language!r}; "
                f"expected one of {sorted(table)}"
            )
        return table[language.lower()]

    def __len__(self):
        return len(self.data_list)

    def _load_audio_to_np(self, x: str) -> Tuple[np.ndarray, int]:

        audio, sr = librosa.load(x, sr=None, mono=True)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        return audio.astype(np.float32), int(sr)

    def _normalize_audio_inputs(self, audios: Union[AudioLike, List[AudioLike]]) -> List[Tuple[np.ndarray, int]]:
        """
        Normalize audio inputs into a list of (waveform, sr).

        Supported forms:
          - str: wav path / URL / base64 audio string
          - np.ndarray: waveform (NOT allowed alone here because sr is unknown)
          - (np.ndarray, sr): waveform + sampling rate
          - list of the above

        Args:
            audios:
                Audio input(s).

        Returns:
            List[Tuple[np.ndarray, int]]:
                List of (float32 waveform, original sr).

        Raises:
            ValueError: If a numpy waveform is provided without sr.
        """
        if isinstance(audios, list):
            items = audios
        else:
            items = [audios]

        out: List[Tuple[np.ndarray, int]] = []
        for a in items:
            if isinstance(a, str):
                out.append(self._load_audio_to_np(a))
            elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
                out.append((a[0].astype(np.float32), int(a[1])))
            elif isinstance(a, np.ndarray):
                raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
            else:
                raise TypeError(f"Unsupported audio input type: {type(a)}")
        return out


    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _ensure_list(self, x: MaybeList) -> List[Any]:
        return x if isinstance(x, list) else [x]

    def _tokenize_texts(self, text) -> List[torch.Tensor]:
        input = self.processor(text=text, return_tensors="pt", padding=True)
        input_id = input["input_ids"]
        input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id
        return input_id

    @torch.inference_mode()
    def extract_mels(self, audio, sr):
        assert sr == 24000, "Only support 24kHz audio"
        mels = mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000
        ).transpose(1, 2)
        return mels



    def __getitem__(self, idx):
        item = self.data_list[idx]

        audio_path  = item["audio"]
        text        = item["text"]
        audio_codes = item["audio_codes"]
        language        = item.get('language','Auto')
        ref_audio_path  = item['ref_audio']
        emotion         = item.get('emotion')

        # Optional so the speaker-only scripts keep working unchanged; the
        # emotion-conditioned trainer validates the field when loading the JSONL.
        if emotion is None:
            emotion_id = -1
        elif emotion in EMOTION_TO_ID:
            emotion_id = EMOTION_TO_ID[emotion]
        else:
            raise ValueError(
                f"Unknown emotion {emotion!r}; expected one of {EMOTIONS}"
            )

        text = self._build_assistant_text(text)
        text_ids = self._tokenize_texts(text)

        audio_codes = torch.tensor(audio_codes, dtype=torch.long)

        ref_audio_list = self._ensure_list(ref_audio_path)
        normalized = self._normalize_audio_inputs(ref_audio_list)
        wav,sr = normalized[0]

        ref_mel = self.extract_mels(audio=wav, sr=sr)

        out = {
            "text_ids": text_ids[:,:-5],    # 1 , t
            "audio_codes":audio_codes,      # t, 16
            "ref_mel":ref_mel,
            "emotion_id":emotion_id,        # index into the emotion table
            # Set by the trainer from the file a row was read from (train/val/test
            # vs replay/extra validation). Rows without it, as in the speaker-only
            # scripts, are all the target speaker.
            "is_target":bool(item.get("is_target", True)),
        }
        if self.language_tag:
            out["language_id"] = self._language_id(language)
        return out

    def collate_fn(self, batch):
        assert self.lag_num == -1

        # p: extra prefix slots for the language id. s: first text slot (8 untagged).
        p = 1 if self.language_tag else 0
        s = 8 + p
        speaker_slot = 6 + p

        item_length = [b['text_ids'].shape[1] + b['audio_codes'].shape[0] for b in batch]
        max_length = max(item_length) + s
        b,t = len(batch),max_length

        input_ids   = torch.zeros((b,t,2),dtype=torch.long)
        codec_ids   = torch.zeros((b,t,16),dtype=torch.long)
        text_embedding_mask     = torch.zeros((b,t),dtype=torch.bool)
        codec_embedding_mask    = torch.zeros((b,t),dtype=torch.bool)
        codec_mask      = torch.zeros((b,t),dtype=torch.bool)
        attention_mask  = torch.zeros((b,t),dtype=torch.long)
        codec_0_labels  = torch.full((b, t), -100, dtype=torch.long)

        for i,data in enumerate(batch):
            text_ids        = data['text_ids']
            audio_codec_0   = data['audio_codes'][:,0]
            audio_codecs    = data['audio_codes']

            text_ids_len = text_ids.shape[1]
            codec_ids_len = audio_codec_0.shape[0]

            # text channel
            input_ids[i,  :3, 0] = text_ids[0,:3]
            input_ids[i, 3:s-1, 0] = self.config.tts_pad_token_id
            input_ids[i,   s-1, 0] = self.config.tts_bos_token_id
            input_ids[i, s:s+text_ids_len-3, 0] = text_ids[0,3:]
            input_ids[i,   s+text_ids_len-3, 0] = self.config.tts_eos_token_id
            input_ids[i, s+text_ids_len-2:s+text_ids_len+codec_ids_len , 0] = self.config.tts_pad_token_id
            text_embedding_mask[i,  :s+text_ids_len+codec_ids_len] = True

            # codec channel
            # input_ids[i,   :3, 1] = 0
            if self.language_tag:
                think_tags = [
                    self.config.talker_config.codec_think_id,
                    self.config.talker_config.codec_think_bos_id,
                    data['language_id'],
                    self.config.talker_config.codec_think_eos_id,
                ]
            else:
                think_tags = [
                    self.config.talker_config.codec_nothink_id,
                    self.config.talker_config.codec_think_bos_id,
                    self.config.talker_config.codec_think_eos_id,
                ]
            input_ids[i,    3:s ,1] = torch.tensor(
                                        think_tags + [
                                            0,     # for speaker embedding
                                            self.config.talker_config.codec_pad_id
                                        ]
                                    )
            input_ids[i,    s:s+text_ids_len-3  ,1] = self.config.talker_config.codec_pad_id
            input_ids[i,    s+text_ids_len-3    ,1] = self.config.talker_config.codec_pad_id
            input_ids[i,    s+text_ids_len-2    ,1] = self.config.talker_config.codec_bos_id
            input_ids[i,    s+text_ids_len-1:s+text_ids_len-1+codec_ids_len,    1] = audio_codec_0
            input_ids[i,    s+text_ids_len-1+codec_ids_len,    1] = self.config.talker_config.codec_eos_token_id

            codec_0_labels[i,    s+text_ids_len-1:s+text_ids_len-1+codec_ids_len] = audio_codec_0
            codec_0_labels[i,    s+text_ids_len-1+codec_ids_len] = self.config.talker_config.codec_eos_token_id

            codec_ids[i, s+text_ids_len-1:s+text_ids_len-1+codec_ids_len,:] = audio_codecs

            codec_embedding_mask[i, 3:s+text_ids_len+codec_ids_len] = True
            codec_embedding_mask[i, speaker_slot] = False       # for speaker embedding

            codec_mask[i,   s+text_ids_len-1:s+text_ids_len-1+codec_ids_len] = True
            attention_mask[i, :s+text_ids_len+codec_ids_len] = True

        ref_mels = [data['ref_mel'] for data in batch]
        ref_mels = torch.cat(ref_mels,dim=0)

        emotion_ids = torch.tensor([data['emotion_id'] for data in batch], dtype=torch.long)
        target_rows = torch.tensor([data.get('is_target', True) for data in batch], dtype=torch.bool)

        return {
            'input_ids':input_ids,
            'ref_mels':ref_mels,
            'emotion_ids':emotion_ids,
            'target_rows':target_rows,
            'attention_mask':attention_mask,
            'text_embedding_mask':text_embedding_mask.unsqueeze(-1),
            'codec_embedding_mask':codec_embedding_mask.unsqueeze(-1),
            'codec_0_labels':codec_0_labels,
            'codec_ids': codec_ids,
            'codec_mask':codec_mask,
            # Where the trainer must write the speaker vector; codec_embedding_mask
            # leaves exactly this slot empty.
            'speaker_slot': speaker_slot,
        }
