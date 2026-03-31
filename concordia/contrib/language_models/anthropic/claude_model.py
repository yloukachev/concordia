# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Language Model that uses Anthropic's Claude models."""

import os
from collections.abc import Collection, Sequence
from typing import override

import anthropic
from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib


_MAX_MULTIPLE_CHOICE_ATTEMPTS = 20


class ClaudeLanguageModel(language_model.LanguageModel):
  """Language Model that uses Anthropic Claude models."""

  def __init__(
      self,
      model_name: str,
      *,
      api_key: str | None = None,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
  ):
    """Initializes the instance.

    Args:
      model_name: The Claude model to use (e.g. 'claude-sonnet-4-20250514').
      api_key: The API key to use. If None, will use the ANTHROPIC_API_KEY
        environment variable.
      measurements: The measurements object to log usage statistics to.
      channel: The channel to write the statistics to.
    """
    if api_key is None:
      api_key = os.getenv('ANTHROPIC_API_KEY')
      if not api_key:
        raise ValueError(
            'ANTHROPIC_API_KEY not found. Please provide it via the api_key '
            'parameter or set the ANTHROPIC_API_KEY environment variable.'
        )
    self._model_name = model_name
    self._client = anthropic.Anthropic(api_key=api_key)
    self._measurements = measurements
    self._channel = channel

  @override
  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      top_p: float = language_model.DEFAULT_TOP_P,
      top_k: int = language_model.DEFAULT_TOP_K,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
  ) -> str:
    del terminators, seed  # Unused for Claude.

    response = self._client.messages.create(
        model=self._model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        system=(
            'You always continue input provided by the user and you never '
            'repeat what the user already said. Be concise.'
        ),
        messages=[
            {
                'role': 'user',
                'content': 'Question: Is Jake a turtle?\nAnswer: Jake is ',
            },
            {'role': 'assistant', 'content': 'not a turtle.'},
            {
                'role': 'user',
                'content': (
                    'Question: What is Priya doing right now?\nAnswer: '
                    'Priya is currently '
                ),
            },
            {'role': 'assistant', 'content': 'sleeping.'},
            {'role': 'user', 'content': prompt},
        ],
        timeout=timeout,
    )

    result = response.content[0].text

    if self._measurements is not None:
      self._measurements.publish_datum(
          self._channel,
          {'raw_text_length': len(result)},
      )

    return result

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, dict[str, float]]:
    del seed  # Unused for Claude.

    prompt = (
        prompt
        + '\nRespond EXACTLY with one of the following strings:\n'
        + '\n'.join(responses)
        + '.'
    )

    for attempts in range(_MAX_MULTIPLE_CHOICE_ATTEMPTS):
      answer = self.sample_text(
          prompt,
          temperature=0.0,
          max_tokens=256,
      )
      answer = answer.strip()
      try:
        idx = responses.index(answer)
      except ValueError:
        # Try partial matching — Claude sometimes adds punctuation.
        for i, r in enumerate(responses):
          if r in answer or answer in r:
            if self._measurements is not None:
              self._measurements.publish_datum(
                  self._channel, {'choices_calls': attempts}
              )
            return i, responses[i], {}
        continue
      else:
        if self._measurements is not None:
          self._measurements.publish_datum(
              self._channel, {'choices_calls': attempts}
          )
        return idx, responses[idx], {}

    raise language_model.InvalidResponseError(
        f'Too many multiple choice attempts.\nLast attempt extracted: {answer}'
    )
