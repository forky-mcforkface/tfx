# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Util functions for contructing input channels."""

from typing import Type

from tfx.types import channel
from tfx.types.artifact import Artifact


def mlmd_query_channel(artifact_type: Type[Artifact], pipeline_name: str,
                       producer_component_id: str):
  """Construct a MLMDQueryChannel instance which queries MLMD to find artifacts.

  Example usage:
    # Gets a MLMDQueryChannel which queries all artifacts of a pipeline named
    # producer-pipeline.
    channel = mlmd_query_chanel(pipeline_name='producer-pipeline')

    # Gets a MLMDQueryChannel which queries all artifacts produced by component
    # ExampleGen in a pipeline named producer-pipeline.
    channel = mlmd_query_chanel(pipeline_name='producer-pipeline',
                                producer_component_id='example_gen')

  Args:
    artifact_type: The type of artifacts this channel returns.
    pipeline_name: Name of a pipeline.
    producer_component_id: ID of a component in the pipeline.

  Returns:
    channel.MLMDQueryChannel instance.
  """
  return channel.MLMDQueryChannel(
      artifact_type=artifact_type,
      pipeline_name=pipeline_name,
      producer_component_id=producer_component_id)
