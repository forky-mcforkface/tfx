# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Experimental Resolver for getting the latest pipeline run artifact."""

from typing import Dict, List, Optional

from tfx import types
from tfx.dsl.compiler import constants
from tfx.dsl.components.common import resolver
from tfx.utils import doc_controls

import ml_metadata as mlmd


class LatestPipelineRunArtifactStrategy(resolver.ResolverStrategy):
  """Strategy that resolves the latest pipeline run artifacts.

  Note that this ResolverStrategy is experimental and is subject to change in
  terms of both interface and implementation.

  Don't construct LatestPipelineRunArtifactStrategy directly, example usage:
  ```
    model_resolver = Resolver(
        strategy_class=LatestPipelineRunArtifactStrategy,
        model=Channel(type=Model),
    ).with_id('latest_model_resolver')
    model_resolver.outputs['model']
  ```
  """

  def __init__(self, desired_num_of_artifacts: Optional[int] = 1):
    self._desired_num_of_artifact = desired_num_of_artifacts

  def _resolve(self, store: mlmd.MetadataStore,
               input_dict: Dict[str, List[types.Artifact]]):
    pipeline_run_contexts = store.get_contexts_by_type(
        constants.PIPELINE_RUN_CONTEXT_TYPE_NAME)
    if not pipeline_run_contexts:
      return {}

    pipeline_run_contexts.sort(key=lambda c: c.id, reverse=True)
    latest_pipeline_run_artifacts = store.get_artifacts_by_context(
        pipeline_run_contexts[0].id)
    latest_pipeline_run_artifact_ids = set(
        [a.id for a in latest_pipeline_run_artifacts])

    result = {}
    for k, artifact_list in input_dict.items():
      selected_artifacts = [
          a for a in artifact_list if a.id in latest_pipeline_run_artifact_ids
      ]
      result[k] = selected_artifacts
    return result

  @doc_controls.do_not_generate_docs
  def resolve_artifacts(
      self, store: mlmd.MetadataStore, input_dict: Dict[str,
                                                        List[types.Artifact]]
  ) -> Optional[Dict[str, List[types.Artifact]]]:
    """Resolves artifacts from channels by querying MLMD.

    Args:
      store: An MLMD MetadataStore object.
      input_dict: The input_dict to resolve from.

    Returns:
      If `min_count` for every input is met, returns a
      Dict[str, List[Artifact]]. Otherwise, return None.
    """
    resolved_dict = self._resolve(store, input_dict)
    all_min_count_met = all(
        len(artifact_list) >= self._desired_num_of_artifact
        for artifact_list in resolved_dict.values())
    return resolved_dict if all_min_count_met else None
