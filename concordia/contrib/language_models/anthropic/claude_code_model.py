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

"""Language Model that uses the Claude Code CLI (no API key needed).

Uses the user's existing Claude Code subscription by shelling out to
`claude -p` for each LLM call.
"""

import subprocess
import shutil
from collections.abc import Collection, Sequence
from typing import override

from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib


_MAX_MULTIPLE_CHOICE_ATTEMPTS = 20
_CLAUDE_SYSTEM_PROMPT = (
    'IMPORTANT: You are NOT a coding assistant right now. You are a text '
    'completion engine for a social simulation. Your ONLY job is to '
    'continue the text the user provides. Do not ask clarifying questions. '
    'Do not mention code, files, or programming. Do not use markdown. '
    'Do not add preamble or commentary. Just continue the text naturally '
    'and concisely as if you are writing the next part of a narrative or '
    'answering the question posed in the prompt.'
)


def _call_claude(prompt: str, system: str = _CLAUDE_SYSTEM_PROMPT,
                 timeout: float = 120) -> str:
  """Call the claude CLI and return the response text."""
  claude_path = shutil.which('claude')
  if not claude_path:
    raise RuntimeError(
        'claude CLI not found. Install Claude Code: '
        'https://docs.anthropic.com/en/docs/claude-code'
    )

  full_prompt = f'{system}\n\n{prompt}' if system else prompt

  result = subprocess.run(
      [claude_path, '-p', full_prompt, '--output-format', 'text'],
      capture_output=True,
      text=True,
      timeout=timeout,
  )

  if result.returncode != 0:
    stderr = result.stderr.strip()
    raise RuntimeError(f'claude CLI error (rc={result.returncode}): {stderr}')

  return result.stdout.strip()


class ClaudeCodeLanguageModel(language_model.LanguageModel):
  """Language Model that uses the Claude Code CLI.

  No API key required — uses the user's Claude Code subscription.
  """

  def __init__(
      self,
      model_name: str = 'claude-code',
      *,
      api_key: str | None = None,  # Accepted but ignored for registry compat.
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
  ):
    del api_key  # Not needed — uses CLI auth.
    self._model_name = model_name
    self._measurements = measurements
    self._channel = channel

    # Verify claude CLI is available at init time.
    if not shutil.which('claude'):
      raise RuntimeError(
          'claude CLI not found on PATH. '
          'Install Claude Code: https://docs.anthropic.com/en/docs/claude-code'
      )

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
    # The CLI doesn't expose temperature/top_p/etc, but that's fine —
    # the model defaults work well for simulation.
    del max_tokens, terminators, temperature, top_p, top_k, seed

    result = _call_claude(prompt, timeout=max(timeout, 120))

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
    del seed

    choice_prompt = (
        prompt
        + '\nRespond EXACTLY with one of the following strings and '
        + 'nothing else:\n'
        + '\n'.join(responses)
    )

    for attempts in range(_MAX_MULTIPLE_CHOICE_ATTEMPTS):
      answer = _call_claude(choice_prompt).strip()

      # Exact match
      try:
        idx = responses.index(answer)
        if self._measurements is not None:
          self._measurements.publish_datum(
              self._channel, {'choices_calls': attempts}
          )
        return idx, responses[idx], {}
      except ValueError:
        pass

      # Fuzzy match — Claude sometimes wraps in quotes or adds punctuation.
      cleaned = answer.strip('"\'.,!? ')
      try:
        idx = responses.index(cleaned)
        if self._measurements is not None:
          self._measurements.publish_datum(
              self._channel, {'choices_calls': attempts}
          )
        return idx, responses[idx], {}
      except ValueError:
        pass

      # Substring match
      for i, r in enumerate(responses):
        if r in answer or answer in r:
          if self._measurements is not None:
            self._measurements.publish_datum(
                self._channel, {'choices_calls': attempts}
            )
          return i, responses[i], {}

    raise language_model.InvalidResponseError(
        f'Too many multiple choice attempts.\nLast attempt: {answer}'
    )
