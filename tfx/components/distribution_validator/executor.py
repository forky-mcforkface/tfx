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
"""TFX DistributionValidator executor."""

import os
from typing import Any, Dict, List, Optional

from absl import logging
import tensorflow_data_validation as tfdv
from tensorflow_data_validation.utils import stats_util
from tfx import types
from tfx.components.statistics_gen import stats_artifact_utils
from tfx.dsl.components.base import base_executor
from tfx.proto import distribution_validator_pb2
from tfx.types import artifact_utils
from tfx.types import standard_component_specs
from tfx.utils import io_utils
from tfx.utils import json_utils

from tensorflow_metadata.proto.v0 import anomalies_pb2
from tensorflow_metadata.proto.v0 import path_pb2
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2

# Default file name for anomalies output.
DEFAULT_FILE_NAME = 'SchemaDiff.pb'

_COMPARISON_ANOMALY_TYPES = frozenset([
    anomalies_pb2.AnomalyInfo.Type.COMPARATOR_CONTROL_DATA_MISSING,
    anomalies_pb2.AnomalyInfo.Type.COMPARATOR_TREATMENT_DATA_MISSING,
    anomalies_pb2.AnomalyInfo.Type.COMPARATOR_L_INFTY_HIGH,
    anomalies_pb2.AnomalyInfo.Type.COMPARATOR_JENSEN_SHANNON_DIVERGENCE_HIGH,
    anomalies_pb2.AnomalyInfo.Type.COMPARATOR_LOW_NUM_EXAMPLES,
    anomalies_pb2.AnomalyInfo.Type.COMPARATOR_HIGH_NUM_EXAMPLES
])


def _get_comparison_only_anomalies(
    anomalies: anomalies_pb2.Anomalies) -> anomalies_pb2.Anomalies:
  """Returns new Anomalies proto with only info from statistics comparison."""
  new_anomalies = anomalies_pb2.Anomalies()
  new_anomalies.baseline.CopyFrom(anomalies.baseline)
  new_anomalies.anomaly_name_format = anomalies.anomaly_name_format
  for each in anomalies.anomaly_info:
    for reason in anomalies.anomaly_info[each].reason:
      if reason.type in _COMPARISON_ANOMALY_TYPES:
        new_reason = new_anomalies.anomaly_info[each].reason.add()
        new_reason.CopyFrom(reason)
        new_anomalies.anomaly_info[each].severity = anomalies.anomaly_info[
            each].severity
  for drift_skew in anomalies.drift_skew_info:
    new_drift_skew = new_anomalies.drift_skew_info.add()
    new_drift_skew.CopyFrom(drift_skew)
  for dataset_reason in anomalies.dataset_anomaly_info.reason:
    if dataset_reason.type in _COMPARISON_ANOMALY_TYPES:
      new_anomalies.dataset_anomaly_info.severity = (
          anomalies.dataset_anomaly_info.severity)
      for dataset_anomaly_reason in anomalies.dataset_anomaly_info.reason:
        new_dataset_reason = new_anomalies.dataset_anomaly_info.reason.add()
        new_dataset_reason.CopyFrom(dataset_anomaly_reason)
  return new_anomalies


def _convert_type(
    stats_type: statistics_pb2.FeatureNameStatistics.Type
) -> schema_pb2.FeatureType:
  """Returns the schema type corresponding to the input statistics type."""
  stats_to_schema_types = {
      statistics_pb2.FeatureNameStatistics.INT: schema_pb2.INT,
      statistics_pb2.FeatureNameStatistics.FLOAT: schema_pb2.FLOAT,
      statistics_pb2.FeatureNameStatistics.STRING: schema_pb2.BYTES,
      statistics_pb2.FeatureNameStatistics.BYTES: schema_pb2.BYTES,
      statistics_pb2.FeatureNameStatistics.STRUCT: schema_pb2.STRUCT
  }
  return stats_to_schema_types[stats_type]


def _get_feature_type_from_statistics(
    statistics: statistics_pb2.DatasetFeatureStatistics,
    feature_path: path_pb2.Path) -> schema_pb2.FeatureType:
  """Returns the type of the specified feature."""
  path = tfdv.FeaturePath.from_proto(feature_path)
  try:
    feature_statistics = tfdv.get_feature_stats(statistics, path)
  except ValueError:
    return schema_pb2.TYPE_UNKNOWN
  return _convert_type(feature_statistics.type)


def _make_schema_from_config(
    config: distribution_validator_pb2.DistributionValidatorConfig,
    statistics_list: statistics_pb2.DatasetFeatureStatisticsList
) -> schema_pb2.Schema:
  """Converts a config to a schema that can be used for data validation."""
  statistics = stats_util.get_default_dataset_statistics(statistics_list)
  schema = schema_pb2.Schema()
  for feature in config.feature:
    new_feature = schema.feature.add()
    for step in feature.path.step[:-1]:
      new_feature.name = step
      new_feature.type = schema_pb2.STRUCT
      new_feature = new_feature.struct_domain.feature.add()
    new_feature.name = feature.path.step[-1]
    new_feature.type = _get_feature_type_from_statistics(
        statistics, feature.path)
    new_feature.drift_comparator.CopyFrom(feature.distribution_comparator)
  if config.HasField('num_examples_comparator'):
    schema.dataset_constraints.num_examples_drift_comparator.CopyFrom(
        config.num_examples_comparator)
  return schema


def _has_feature_stats(path: path_pb2.Path,
                       stats: statistics_pb2.DatasetFeatureStatistics) -> bool:
  """Specifies whether stats contain the feature specified by path."""
  for feature_stats in stats.features:
    if feature_stats.path == path:
      return True
  return False


def _check_for_missing_data(
    test_stats: statistics_pb2.DatasetFeatureStatisticsList,
    baseline_stats: statistics_pb2.DatasetFeatureStatisticsList,
    config: distribution_validator_pb2.DistributionValidatorConfig
) -> Optional[anomalies_pb2.Anomalies]:
  """Identifies whether the data needed for distribution validation is missing.

  Note this function does not generate an anomaly where the baseline stats are
  present but are missing statistics for a feature to be validated. That case
  is covered by tfdv.validate_statistics.

  Args:
    test_stats: The test statistics to be analyzed for missing data.
    baseline_stats: The baseline statistics to be analyzed for missing data.
    config: The config that identifies the features for which distribution
      validation will be done.
  Returns:
    If data is missing, returns an Anomalies proto identifying the missing
    data. If data is not missing, returns None.
  """
  test = stats_util.get_default_dataset_statistics(test_stats)
  base = stats_util.get_default_dataset_statistics(baseline_stats)
  anomalies = anomalies_pb2.Anomalies()
  if test.num_examples == 0:
    for feature in config.feature:
      feature_key = '.'.join(feature.path.step)
      reason = anomalies.anomaly_info[feature_key].reason.add()
      reason.type = (
          anomalies_pb2.AnomalyInfo.Type.COMPARATOR_TREATMENT_DATA_MISSING)
      reason.short_description = 'No test data found.'
      anomalies.anomaly_info[
          feature_key].severity = anomalies_pb2.AnomalyInfo.Severity.ERROR
  else:
    for feature in config.feature:
      if not _has_feature_stats(feature.path, test):
        feature_key = '.'.join(feature.path.step)
        reason = anomalies.anomaly_info[feature_key].reason.add()
        reason.type = (
            anomalies_pb2.AnomalyInfo.Type.COMPARATOR_TREATMENT_DATA_MISSING)
        reason.short_description = 'Feature not found in test data.'
        anomalies.anomaly_info[
            feature_key].severity = anomalies_pb2.AnomalyInfo.Severity.ERROR
  if base.num_examples == 0:
    for feature in config.feature:
      feature_key = '.'.join(feature.path.step)
      reason = anomalies.anomaly_info[feature_key].reason.add()
      reason.type = (
          anomalies_pb2.AnomalyInfo.Type.COMPARATOR_CONTROL_DATA_MISSING)
      reason.short_description = 'No baseline data found.'
      anomalies.anomaly_info[
          feature_key].severity = anomalies_pb2.AnomalyInfo.Severity.ERROR
  if anomalies.anomaly_info:
    anomalies.baseline.CopyFrom(
        _make_schema_from_config(config, baseline_stats))
    anomalies.anomaly_name_format = anomalies_pb2.Anomalies.SERIALIZED_PATH
    return anomalies
  return None


class Executor(base_executor.BaseExecutor):
  """DistributionValidator component executor."""

  def Do(self, input_dict: Dict[str, List[types.Artifact]],
         output_dict: Dict[str, List[types.Artifact]],
         exec_properties: Dict[str, Any]) -> None:
    """DistributionValidator executor entrypoint.

    This checks for changes in data distributions from one dataset to another,
    based on the summary statitics for those datasets.

    Args:
      input_dict: Input dict from input key to a list of artifacts.
      output_dict: Output dict from key to a list of artifacts.
      exec_properties: A dict of execution properties.

    Returns:
      None
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    # Load and deserialize include splits from execution properties.
    include_splits_list = json_utils.loads(
        exec_properties.get(standard_component_specs.INCLUDE_SPLIT_PAIRS_KEY,
                            'null')) or []
    include_splits = set((test, base) for test, base in include_splits_list)

    test_statistics = artifact_utils.get_single_instance(
        input_dict[standard_component_specs.STATISTICS_KEY])
    baseline_statistics = artifact_utils.get_single_instance(
        input_dict[standard_component_specs.BASELINE_STATISTICS_KEY])

    config = exec_properties.get(
        standard_component_specs.DISTRIBUTION_VALIDATOR_CONFIG_KEY)

    logging.info('Running distribution_validator with config %s', config)

    # Set up pairs of splits to validate.
    split_pairs = []
    for test_split in artifact_utils.decode_split_names(
        test_statistics.split_names):
      for baseline_split in artifact_utils.decode_split_names(
          baseline_statistics.split_names):
        if not include_splits or (test_split, baseline_split) in include_splits:
          split_pairs.append((test_split, baseline_split))

    anomalies_artifact = artifact_utils.get_single_instance(
        output_dict[standard_component_specs.ANOMALIES_KEY])
    anomalies_artifact.split_names = artifact_utils.encode_split_names(
        ['%s_%s' % (test, baseline) for test, baseline in split_pairs])

    schema = None
    for test_split, baseline_split in split_pairs:
      split_pair = '%s_%s' % (test_split, baseline_split)
      logging.info('Processing split pair %s', split_pair)
      test_stats_split = stats_artifact_utils.load_statistics(
          test_statistics, test_split).proto()
      baseline_stats_split = stats_artifact_utils.load_statistics(
          baseline_statistics, baseline_split).proto()

      # If one or both is missing data, raise a data missing anomaly. This
      # anomaly will apply where the entire dataset is missing and where the
      # dataset is present but it is missing statistics for a relevant
      # feature(s). TFDV covers only the case where the baseline statistics are
      # present but missing a relevant feature in its validation, so
      # _check_for_missing_data covers the rest.
      maybe_missing_anomalies = _check_for_missing_data(test_stats_split,
                                                        baseline_stats_split,
                                                        config)
      if maybe_missing_anomalies is not None:
        anomalies = maybe_missing_anomalies
      else:
        if schema is None:
          schema = _make_schema_from_config(config, baseline_stats_split)
        full_anomalies = tfdv.validate_statistics(
            test_stats_split, schema, previous_statistics=baseline_stats_split)
        anomalies = _get_comparison_only_anomalies(full_anomalies)
      io_utils.write_bytes_file(
          os.path.join(anomalies_artifact.uri, 'SplitPair-%s' % split_pair,
                       DEFAULT_FILE_NAME), anomalies.SerializeToString())
