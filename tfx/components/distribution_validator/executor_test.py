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
"""Tests for tfx.distribution_validator.executor."""

import os
import tempfile

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from tfx.components.distribution_validator import executor
from tfx.dsl.io import fileio
from tfx.proto import distribution_validator_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx.utils import io_utils
from tfx.utils import json_utils
from tensorflow_metadata.proto.v0 import anomalies_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2

from google.protobuf import text_format

FLAGS = flags.FLAGS


class ExecutorTest(parameterized.TestCase):

  def get_temp_dir(self):
    return tempfile.mkdtemp()

  @parameterized.named_parameters(
      {
          'testcase_name': 'split_pairs_specified',
          'split_pairs': [('train', 'eval')],
          'expected_split_pair_names': {'train_eval'},
      }, {
          'testcase_name': 'implicit_split_pairs',
          'split_pairs': None,
          'expected_split_pair_names':
              {'train_train', 'train_eval', 'eval_eval', 'eval_train'},
      })
  def testSplitPairs(self, split_pairs, expected_split_pair_names):
    source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')

    stats_artifact = standard_artifacts.ExampleStatistics()
    stats_artifact.uri = os.path.join(source_data_dir, 'statistics_gen')
    stats_artifact.split_names = artifact_utils.encode_split_names(
        ['train', 'eval'])

    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    validation_output = standard_artifacts.ExampleAnomalies()
    validation_output.uri = os.path.join(output_data_dir, 'output')

    input_dict = {
        standard_component_specs.STATISTICS_KEY: [stats_artifact],
        standard_component_specs.BASELINE_STATISTICS_KEY: [stats_artifact],
    }

    exec_properties = {
        # List needs to be serialized before being passed into Do function.
        standard_component_specs.INCLUDE_SPLIT_PAIRS_KEY:
            json_utils.dumps(split_pairs),
        standard_component_specs.DISTRIBUTION_VALIDATOR_CONFIG_KEY:
            distribution_validator_pb2.DistributionValidatorConfig(),
    }

    output_dict = {
        standard_component_specs.ANOMALIES_KEY: [validation_output],
    }

    distribution_validator_executor = executor.Executor()
    distribution_validator_executor.Do(input_dict, output_dict, exec_properties)

    for split_pair_name in expected_split_pair_names:
      output_path = os.path.join(validation_output.uri,
                                 'SplitPair-' + split_pair_name,
                                 'SchemaDiff.pb')
      self.assertTrue(fileio.exists(output_path))

    # Confirm that no unexpected result files exist.
    all_outputs = fileio.glob(
        os.path.join(validation_output.uri, 'SplitPair-*'))
    for output in all_outputs:
      split_pair = output.split('SplitPair-')[1]
      self.assertIn(split_pair, expected_split_pair_names)

  @parameterized.named_parameters(
      {
          'testcase_name':
              'multiple_features',
          'config':
              """
              feature: {
                  path: {
                      step: 'company'
                  }
                  distribution_comparator: {
                    infinity_norm: {
                        threshold: 0.0
                    }
                  }
              }
              feature: {
                  path: {
                      step: 'dropoff_census_tract'
                  }
                  distribution_comparator: {
                    jensen_shannon_divergence: {
                        threshold: 0.0
                    }
                  }
              }
            """,
          'expected_anomalies':
              """
        baseline {
          feature {
            name: "company"
            type: BYTES
            drift_comparator {
              infinity_norm {
                threshold: 0.0
              }
            }
          }
          feature {
            name: "dropoff_census_tract"
            type: INT
            drift_comparator {
              jensen_shannon_divergence {
                threshold: 0.0
              }
            }
          }
        }
        anomaly_info {
          key: "company"
          value {
            severity: ERROR
            reason {
              type: COMPARATOR_L_INFTY_HIGH
              short_description: "High Linfty distance between current and previous"
              description: "The Linfty distance between current and previous is 0.0122771 (up to six significant digits), above the threshold 0. The feature value with maximum difference is: Dispatch Taxi Affiliation"
            }
          }
        }
        anomaly_info {
          key: "dropoff_census_tract"
          value {
            severity: ERROR
            reason {
              type: COMPARATOR_JENSEN_SHANNON_DIVERGENCE_HIGH
              short_description: "High approximate Jensen-Shannon divergence between current and previous"
              description: "The approximate Jensen-Shannon divergence between current and previous is 0.000917363 (up to six significant digits), above the threshold 0."
            }
          }
        }
        anomaly_name_format: SERIALIZED_PATH
        drift_skew_info {
          path {
            step: "company"
          }
          drift_measurements {
            type: L_INFTY
            value: 0.012277129468474923
            threshold: 0.0
          }
        }
        drift_skew_info {
          path {
            step: "dropoff_census_tract"
          }
          drift_measurements {
            type: JENSEN_SHANNON_DIVERGENCE
            value: 0.000917362998174601
            threshold: 0.0
          }
        }
          """,
      }, {
          'testcase_name':
              'dataset_constraint',
          'config':
              """
          num_examples_comparator: {
              min_fraction_threshold: 1.0,
              max_fraction_threshold: 1.0
          }
       """,
          'expected_anomalies':
              """baseline {
                  dataset_constraints {
                    num_examples_drift_comparator {
                      min_fraction_threshold: 1.0
                      max_fraction_threshold: 1.0
                    }
                  }
                }
                anomaly_name_format: SERIALIZED_PATH
                dataset_anomaly_info {
                  severity: ERROR
                  reason {
                    type: COMPARATOR_HIGH_NUM_EXAMPLES
                    short_description: "High num examples in current dataset versus the previous span."
                    description: "The ratio of num examples in the current dataset versus the previous span is 2.02094 (up to six significant digits), which is above the threshold 1."
                  }
                }""",
      })
  def testAnomaliesGenerated(self, config, expected_anomalies):
    source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')

    stats_artifact = standard_artifacts.ExampleStatistics()
    stats_artifact.uri = os.path.join(source_data_dir, 'statistics_gen')
    stats_artifact.split_names = artifact_utils.encode_split_names(
        ['train', 'eval'])

    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    validation_output = standard_artifacts.ExampleAnomalies()
    validation_output.uri = os.path.join(output_data_dir, 'output')

    validation_config = text_format.Parse(
        config, distribution_validator_pb2.DistributionValidatorConfig())

    input_dict = {
        standard_component_specs.STATISTICS_KEY: [stats_artifact],
        standard_component_specs.BASELINE_STATISTICS_KEY: [stats_artifact],
    }

    # The analyzed splits are set for this test to get a single result proto.
    exec_properties = {
        # List needs to be serialized before being passed into Do function.
        standard_component_specs.INCLUDE_SPLIT_PAIRS_KEY:
            json_utils.dumps([('train', 'eval')]),
        standard_component_specs.DISTRIBUTION_VALIDATOR_CONFIG_KEY:
            validation_config,
    }

    output_dict = {
        standard_component_specs.ANOMALIES_KEY: [validation_output],
    }

    distribution_validator_executor = executor.Executor()
    distribution_validator_executor.Do(input_dict, output_dict, exec_properties)

    self.assertEqual(
        artifact_utils.encode_split_names(['train_eval']),
        validation_output.split_names)

    distribution_anomalies_path = os.path.join(validation_output.uri,
                                               'SplitPair-train_eval',
                                               'SchemaDiff.pb')
    self.assertTrue(fileio.exists(distribution_anomalies_path))
    distribution_anomalies_bytes = io_utils.read_bytes_file(
        distribution_anomalies_path)
    distribution_anomalies = anomalies_pb2.Anomalies()
    distribution_anomalies.ParseFromString(distribution_anomalies_bytes)
    expected_anomalies = text_format.Parse(expected_anomalies,
                                           anomalies_pb2.Anomalies())
    self.assertEqual(expected_anomalies, distribution_anomalies)

  def testStructData(self):
    source_data_dir = FLAGS.test_tmpdir
    stats_artifact = standard_artifacts.ExampleStatistics()
    stats_artifact.uri = os.path.join(source_data_dir, 'statistics_gen')
    stats_artifact.split_names = artifact_utils.encode_split_names(
        ['train', 'eval'])

    struct_stats_train = text_format.Parse(
        """
      datasets {
        num_examples: 12
        features {
          path {
            step: "parent_feature"
          }
          type: STRUCT
          struct_stats {
            common_stats {
                num_non_missing: 12
            }
          }
        }
        features {
          path {
            step: "parent_feature"
            step: "value_feature"
          }
          type: INT
          num_stats {
            common_stats {
                num_non_missing: 12
            }
            histograms {
              buckets {
                low_value: 0.0
                high_value: 9.0
                sample_count: 12.0
              }
              type: STANDARD
            }
          }
        }
      }
      """, statistics_pb2.DatasetFeatureStatisticsList())

    struct_stats_eval = text_format.Parse(
        """
        datasets {
          num_examples: 12
          features {
            path {
              step: "parent_feature"
            }
            type: STRUCT
            struct_stats {
              common_stats {
                  num_non_missing: 12
              }
            }
          }
          features {
            path {
              step: "parent_feature"
              step: "value_feature"
            }
            type: INT
            num_stats {
              common_stats {
                  num_non_missing: 12
              }
              histograms {
                buckets {
                  low_value: 10.0
                  high_value: 19.0
                  sample_count: 12.0
                }
                type: STANDARD
              }
            }
          }
        }
        """, statistics_pb2.DatasetFeatureStatisticsList())

    validation_config = text_format.Parse(
        """
      feature: {
          path: {
              step: 'parent_feature'
              step: 'value_feature'
          }
          distribution_comparator: {
            jensen_shannon_divergence: {
                threshold: 0.0
            }
          }
      }""", distribution_validator_pb2.DistributionValidatorConfig())

    expected_anomalies = text_format.Parse(
        """
      baseline {
        feature {
          name: "parent_feature"
          type: STRUCT
          struct_domain {
            feature {
              name: "value_feature"
              type: INT
              drift_comparator {
                jensen_shannon_divergence {
                  threshold: 0.0
                }
              }
            }
          }
        }
      }
      anomaly_info {
        key: "parent_feature.value_feature"
        value {
          severity: ERROR
          reason {
            type: COMPARATOR_JENSEN_SHANNON_DIVERGENCE_HIGH
            short_description: "High approximate Jensen-Shannon divergence between current and previous"
            description: "The approximate Jensen-Shannon divergence between current and previous is 1 (up to six significant digits), above the threshold 0."
          }
        }
      }
      anomaly_name_format: SERIALIZED_PATH
      drift_skew_info {
        path {
          step: "parent_feature"
          step: "value_feature"
        }
        drift_measurements {
          type: JENSEN_SHANNON_DIVERGENCE
          value: 1.0
          threshold: 0.0
        }
      }""", anomalies_pb2.Anomalies())

    # Create stats artifacts with a struct feature.
    for split_dir in ['Split-eval', 'Split-train']:
      full_split_dir = os.path.join(stats_artifact.uri, split_dir)
      fileio.makedirs(full_split_dir)
      stats_path = os.path.join(full_split_dir, 'FeatureStats.pb')
      if split_dir == 'Split-eval':
        io_utils.write_bytes_file(stats_path,
                                  struct_stats_eval.SerializeToString())
      else:
        io_utils.write_bytes_file(stats_path,
                                  struct_stats_train.SerializeToString())

    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    validation_output = standard_artifacts.ExampleAnomalies()
    validation_output.uri = os.path.join(output_data_dir, 'output')

    input_dict = {
        standard_component_specs.STATISTICS_KEY: [stats_artifact],
        standard_component_specs.BASELINE_STATISTICS_KEY: [stats_artifact],
    }

    # The analyzed splits are set for this test to get a single result proto.
    exec_properties = {
        # List needs to be serialized before being passed into Do function.
        standard_component_specs.INCLUDE_SPLIT_PAIRS_KEY:
            json_utils.dumps([('train', 'eval')]),
        standard_component_specs.DISTRIBUTION_VALIDATOR_CONFIG_KEY:
            validation_config,
    }

    output_dict = {
        standard_component_specs.ANOMALIES_KEY: [validation_output],
    }

    distribution_validator_executor = executor.Executor()
    distribution_validator_executor.Do(input_dict, output_dict, exec_properties)

    self.assertEqual(
        artifact_utils.encode_split_names(['train_eval']),
        validation_output.split_names)

    distribution_anomalies_path = os.path.join(validation_output.uri,
                                               'SplitPair-train_eval',
                                               'SchemaDiff.pb')
    self.assertTrue(fileio.exists(distribution_anomalies_path))
    distribution_anomalies_bytes = io_utils.read_bytes_file(
        distribution_anomalies_path)
    distribution_anomalies = anomalies_pb2.Anomalies()
    distribution_anomalies.ParseFromString(distribution_anomalies_bytes)
    self.assertEqual(expected_anomalies, distribution_anomalies)

  @parameterized.named_parameters(
      {
          'testcase_name':
              'missing_test',
          'stats_train':
              """
            datasets {
              num_examples: 0
            }
          """,
          'stats_eval':
              """
          datasets {
            num_examples: 10
            features {
              path {
                step: "first_feature"
              }
              type: INT
              num_stats {
                common_stats {
                    num_non_missing: 10
                }
                histograms {
                  buckets {
                    low_value: 10.0
                    high_value: 19.0
                    sample_count: 10.0
                  }
                  type: STANDARD
                }
              }
            }
          }
          """,
          'expected_anomalies':
              """
              baseline {
              feature {
                name: "first_feature"
                type: INT
                drift_comparator {
                  jensen_shannon_divergence {
                    threshold: 0.0
                  }
                }
              }
            }
            anomaly_name_format: SERIALIZED_PATH
            anomaly_info {
              key: "first_feature"
              value {
                severity: ERROR
                reason {
                  type: COMPARATOR_TREATMENT_DATA_MISSING
                  short_description: "No test data found."
                }
              }
            }
          """
      }, {
          'testcase_name':
              'missing_baseline',
          'stats_train':
              """
          datasets {
            num_examples: 10
            features {
              path {
                step: "first_feature"
              }
              type: INT
              num_stats {
                common_stats {
                    num_non_missing: 10
                }
                histograms {
                  buckets {
                    low_value: 10.0
                    high_value: 19.0
                    sample_count: 10.0
                  }
                  type: STANDARD
                }
              }
            }
          }""",
          'stats_eval':
              """
            datasets {
              num_examples: 0
            }
          """,
          'expected_anomalies':
              """
            baseline {
              feature {
                name: "first_feature"
                type: TYPE_UNKNOWN
                drift_comparator {
                  jensen_shannon_divergence {
                    threshold: 0.0
                  }
                }
              }
            }
            anomaly_name_format: SERIALIZED_PATH
            anomaly_info {
              key: "first_feature"
              value {
                severity: ERROR
                reason {
                  type: COMPARATOR_CONTROL_DATA_MISSING
                  short_description: "No baseline data found."
                }
              }
            }
          """
      }, {
          'testcase_name':
              'missing_test_feature',
          'stats_train':
              """
          datasets {
            num_examples: 10
            features {
              path {
                step: "other_feature"
              }
              type: INT
              num_stats {
                common_stats {
                    num_non_missing: 10
                }
                histograms {
                  buckets {
                    low_value: 10.0
                    high_value: 19.0
                    sample_count: 10.0
                  }
                  type: STANDARD
                }
              }
            }
          }""",
          'stats_eval':
              """
          datasets {
            num_examples: 10
            features {
              path {
                step: "first_feature"
              }
              type: INT
              num_stats {
                common_stats {
                    num_non_missing: 10
                }
                histograms {
                  buckets {
                    low_value: 10.0
                    high_value: 19.0
                    sample_count: 10.0
                  }
                  type: STANDARD
                }
              }
            }
          }""",
          'expected_anomalies':
              """
            baseline {
              feature {
                name: "first_feature"
                type: INT
                drift_comparator {
                  jensen_shannon_divergence {
                    threshold: 0.0
                  }
                }
              }
            }
            anomaly_name_format: SERIALIZED_PATH
            anomaly_info {
              key: "first_feature"
              value {
                severity: ERROR
                reason {
                  type: COMPARATOR_TREATMENT_DATA_MISSING
                  short_description: "Feature not found in test data."
                }
              }
            }
          """
      }, {
          'testcase_name':
              'missing_baseline_feature',
          'stats_train':
              """
          datasets {
            num_examples: 10
            features {
              path {
                step: "first_feature"
              }
              type: INT
              num_stats {
                common_stats {
                    num_non_missing: 10
                }
                histograms {
                  buckets {
                    low_value: 10.0
                    high_value: 19.0
                    sample_count: 10.0
                  }
                  type: STANDARD
                }
              }
            }
          }""",
          'stats_eval':
              """
          datasets {
            num_examples: 10
            features {
              path {
                step: "other_feature"
              }
              type: INT
              num_stats {
                common_stats {
                    num_non_missing: 10
                }
                histograms {
                  buckets {
                    low_value: 10.0
                    high_value: 19.0
                    sample_count: 10.0
                  }
                  type: STANDARD
                }
              }
            }
          }""",
          'expected_anomalies':
              """
            baseline {
              feature {
                name: "first_feature"
                type: TYPE_UNKNOWN
                drift_comparator {
                  jensen_shannon_divergence {
                    threshold: 0.0
                  }
                }
              }
            }
            anomaly_name_format: SERIALIZED_PATH
            anomaly_info {
              key: "first_feature"
              value {
                severity: ERROR
                reason {
                  type: COMPARATOR_CONTROL_DATA_MISSING
                  short_description: "previous data missing"
                  description: "previous data is missing."
                }
              }
            }
          """
      })
  def testEmptyData(self, stats_train, stats_eval, expected_anomalies):
    source_data_dir = FLAGS.test_tmpdir
    stats_artifact = standard_artifacts.ExampleStatistics()
    stats_artifact.uri = os.path.join(source_data_dir, 'statistics_gen')
    stats_artifact.split_names = artifact_utils.encode_split_names(
        ['train', 'eval'])

    validation_config = text_format.Parse(
        """
        feature: {
            path: {
                step: 'first_feature'
            }
            distribution_comparator: {
              jensen_shannon_divergence: {
                  threshold: 0.0
              }
            }
        }
        """, distribution_validator_pb2.DistributionValidatorConfig())

    train_stats = text_format.Parse(
        stats_train, statistics_pb2.DatasetFeatureStatisticsList())
    eval_stats = text_format.Parse(
        stats_eval, statistics_pb2.DatasetFeatureStatisticsList())

    # Create stats artifacts with a struct feature.
    for split_dir in ['Split-eval', 'Split-train']:
      full_split_dir = os.path.join(stats_artifact.uri, split_dir)
      fileio.makedirs(full_split_dir)
      stats_path = os.path.join(full_split_dir, 'FeatureStats.pb')
      if split_dir == 'Split-eval':
        io_utils.write_bytes_file(stats_path, eval_stats.SerializeToString())
      else:
        io_utils.write_bytes_file(stats_path, train_stats.SerializeToString())

    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    validation_output = standard_artifacts.ExampleAnomalies()
    validation_output.uri = os.path.join(output_data_dir, 'output')

    input_dict = {
        standard_component_specs.STATISTICS_KEY: [stats_artifact],
        standard_component_specs.BASELINE_STATISTICS_KEY: [stats_artifact],
    }

    # The analyzed splits are set for this test to get a single result proto.
    exec_properties = {
        # List needs to be serialized before being passed into Do function.
        standard_component_specs.INCLUDE_SPLIT_PAIRS_KEY:
            json_utils.dumps([('train', 'eval')]),
        standard_component_specs.DISTRIBUTION_VALIDATOR_CONFIG_KEY:
            validation_config,
    }

    output_dict = {
        standard_component_specs.ANOMALIES_KEY: [validation_output],
    }

    distribution_validator_executor = executor.Executor()
    distribution_validator_executor.Do(input_dict, output_dict, exec_properties)

    self.assertEqual(
        artifact_utils.encode_split_names(['train_eval']),
        validation_output.split_names)

    expected_anomalies = text_format.Parse(expected_anomalies,
                                           anomalies_pb2.Anomalies())
    distribution_anomalies_path = os.path.join(validation_output.uri,
                                               'SplitPair-train_eval',
                                               'SchemaDiff.pb')
    self.assertTrue(fileio.exists(distribution_anomalies_path))
    distribution_anomalies_bytes = io_utils.read_bytes_file(
        distribution_anomalies_path)
    distribution_anomalies = anomalies_pb2.Anomalies()
    distribution_anomalies.ParseFromString(distribution_anomalies_bytes)
    self.assertEqual(expected_anomalies, distribution_anomalies)


if __name__ == '__main__':
  absltest.main()
