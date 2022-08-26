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
"""Test for LatestArtifactStrategy."""

import tensorflow as tf
from tfx.dsl.input_resolution.strategies import latest_pipeline_run_artifact_strategy as strategy
from tfx.orchestration import metadata
from tfx.orchestration.portable.mlmd import common_utils
from tfx.types import standard_artifacts
from tfx.utils import test_case_utils

from ml_metadata.proto import metadata_store_pb2


class LatestPipelineRunStrategyTest(test_case_utils.TfxTest):

  def setUp(self):
    super().setUp()
    self._connection_config = metadata_store_pb2.ConnectionConfig()
    self._connection_config.sqlite.SetInParent()
    self._metadata = self.enter_context(
        metadata.Metadata(connection_config=self._connection_config))
    self._store = self._metadata.store

  def testStrategy(self):
    # Prepare artifacts.
    input_artifact_1 = standard_artifacts.Examples()
    input_artifact_1.uri = 'example'
    input_artifact_1.type_id = common_utils.register_type_if_not_exist(
        self._metadata, input_artifact_1.artifact_type).id
    input_artifact_2 = standard_artifacts.Examples()
    input_artifact_2.uri = 'example'
    input_artifact_2.type_id = common_utils.register_type_if_not_exist(
        self._metadata, input_artifact_2.artifact_type).id
    [input_artifact_1.id,
     input_artifact_2.id] = self._metadata.store.put_artifacts(
         [input_artifact_1.mlmd_artifact, input_artifact_2.mlmd_artifact])

    # Prepare contexts.
    context_type = metadata_store_pb2.ContextType()
    context_type.name = 'pipeline_run'
    context_type_id = self._metadata.store.put_context_type(context_type)
    pipeline_run_context_1 = metadata_store_pb2.Context()
    pipeline_run_context_1.type_id = context_type_id
    pipeline_run_context_1.name = 'run-20220825-175117-371960'
    [pipeline_run_context_1.id
    ] = self._metadata.store.put_contexts([pipeline_run_context_1])
    pipeline_run_context_2 = metadata_store_pb2.Context()
    pipeline_run_context_2.type_id = context_type_id
    pipeline_run_context_2.name = 'run-20220825-175117-371961'
    [pipeline_run_context_2.id
    ] = self._metadata.store.put_contexts([pipeline_run_context_2])
    attribution_1 = metadata_store_pb2.Attribution()
    attribution_1.artifact_id = input_artifact_1.id
    attribution_1.context_id = pipeline_run_context_1.id
    attribution_2 = metadata_store_pb2.Attribution()
    attribution_2.artifact_id = input_artifact_2.id
    attribution_2.context_id = pipeline_run_context_2.id
    self._metadata.store.put_attributions_and_associations(
        [attribution_1, attribution_2], [])

    # Run the function for test.
    latest_pipeline_run_strategy = strategy.LatestPipelineRunArtifactStrategy()
    result = latest_pipeline_run_strategy.resolve_artifacts(
        self._store, {'input': [input_artifact_1, input_artifact_2]})

    # Test the results.
    expected_artifact = [input_artifact_2]
    self.assertIsNotNone(result)
    self.assertEqual(expected_artifact, result['input'])


if __name__ == '__main__':
  tf.test.main()
