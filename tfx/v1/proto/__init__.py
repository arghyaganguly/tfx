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
"""TFX proto module."""

from tfx.proto import bulk_inferrer_pb2
from tfx.proto import evaluator_pb2
from tfx.proto import example_gen_pb2
from tfx.proto import infra_validator_pb2
from tfx.proto import pusher_pb2
from tfx.proto import range_config_pb2
from tfx.proto import trainer_pb2
from tfx.proto import transform_pb2
from tfx.proto import tuner_pb2

ModelSpec = bulk_inferrer_pb2.ModelSpec
DataSpec = bulk_inferrer_pb2.DataSpec
OutputExampleSpec = bulk_inferrer_pb2.OutputExampleSpec
OutputColumnsSpec = bulk_inferrer_pb2.OutputColumnsSpec
ClassifyOutput = bulk_inferrer_pb2.ClassifyOutput
RegressOutput = bulk_inferrer_pb2.RegressOutput
PredictOutput = bulk_inferrer_pb2.PredictOutput
PredictOutputCol = bulk_inferrer_pb2.PredictOutputCol
del bulk_inferrer_pb2

ModelSpec.__doc__ = """
Specifies the signature name to run the inference in BulkInferrer.
"""

DataSpec.__doc__ = """
Indicates which splits of examples should be processed in BulkInferrer.
"""

OutputExampleSpec.__doc__ = """
Defines how the inferrence results map to columns in output example in BulkInferrer.
"""

OutputColumnsSpec.__doc__ = """
The signature_name should exist in `ModelSpec.model_signature_name`. 
You can leave it unset if no more than one `ModelSpec.model_signature_name` is 
specified in your bulk inferrer.
"""

ClassifyOutput.__doc__ = """
One type of output_type under OutputColumnsSpec.
"""

RegressOutput.__doc__ = """
One type of output_type under OutputColumnsSpec.
"""

PredictOutput.__doc__ = """
One type of output_type under OutputColumnsSpec.
"""

PredictOutputCol.__doc__ = """
Proto type of output_columns under PredictOutput.
"""

FeatureSlicingSpec = evaluator_pb2.FeatureSlicingSpec
SingleSlicingSpec = evaluator_pb2.SingleSlicingSpec
del evaluator_pb2

FeatureSlicingSpec.__doc__ = """
Slices corresponding to data set in Evaluator.
"""

SingleSlicingSpec.__doc__ = """
Specifies a single directive for choosing features for slicing.
An empty proto means we do not slice on features (i.e. use the entire data set).
"""

CustomConfig = example_gen_pb2.CustomConfig
Input = example_gen_pb2.Input
Output = example_gen_pb2.Output
SplitConfig = example_gen_pb2.SplitConfig
PayloadFormat = example_gen_pb2.PayloadFormat
del example_gen_pb2

CustomConfig.__doc__ = """
Optional specified configuration for ExampleGen.
"""

Input.__doc__ = """
Specification of the input of ExampleGen.
"""

# Split.__doc__ = """
# List of split name and input glob pattern pairs in Input of ExampleGen.
# 'name' shouldn't be empty and must be unique within the list.
# 'pattern' is a glob file pattern that maps to input files. There are
# some specs that map the data into TFX data hierarchy.  Some ExampleGen's
# might take the pattern as something other than filepatterns (e.g. SQL
# queries for DremelToExample).
# """

Output.__doc__ = """
Specification of the output of the ExampleGen.
"""

SplitConfig.__doc__ = """
A config to partition examples into split in ExampleGen.
"""

PayloadFormat.__doc__ = """
Enum to indicate payload format that ExampleGen produces.
"""

ServingSpec = infra_validator_pb2.ServingSpec
ValidationSpec = infra_validator_pb2.ValidationSpec
TensorFlowServing = infra_validator_pb2.TensorFlowServing
LocalDockerConfig = infra_validator_pb2.LocalDockerConfig
KubernetesConfig = infra_validator_pb2.KubernetesConfig
RequestSpec = infra_validator_pb2.RequestSpec
TensorFlowServingRequestSpec = infra_validator_pb2.TensorFlowServingRequestSpec
del infra_validator_pb2

ServingSpec.__doc__ = """
ServingSpec defines an environment of the validating infrastructure in InfraValidator.
"""

ValidationSpec.__doc__ = """
Specification for validation criteria and thresholds in InfraValidator.
"""

TensorFlowServing.__doc__ = """
TensorFlow Serving docker image (tensorflow/serving) for serving binary.
"""

LocalDockerConfig.__doc__ = """
Docker runtime in a local machine. This is useful when you're running pipeline with infra validator component in your your local machine. 
You need to install docker in advance.
"""

KubernetesConfig.__doc__ = """
Kubernetes configuration. We currently only support the use case when infra validator is run by KubeflowDagRunner.
Model server will be launched in the same namespace KFP is running on, as well as same service account will be
used (unless specified).
Model server will have ownerReferences to the infra validator, which
delegates the strict cleanup guarantee to the kubernetes cluster.
"""

RequestSpec.__doc__ = """
Optional configuration about making requests from examples input in InfraValidator.
"""

TensorFlowServingRequestSpec.__doc__ = """
Request spec for building TF Serving requests.
"""

PushDestination = pusher_pb2.PushDestination
Versioning = pusher_pb2.Versioning
Filesystem = pusher_pb2.PushDestination.Filesystem
del pusher_pb2

PushDestination.__doc__ = """
Defines the destination of pusher in Pusher.
"""

Versioning.__doc__ = """
Versioning method for the model to be pushed. Note that This is the semantic
TFX provides, therefore depending on the platform, some versioning method
might not be compatible. For example TF Serving only accepts an integer
version that is monotonically increasing.
"""

Filesystem.__doc__ = """
File system based destination definition.
"""

RangeConfig = range_config_pb2.RangeConfig
RollingRange = range_config_pb2.RollingRange
StaticRange = range_config_pb2.StaticRange
del range_config_pb2

RangeConfig.__doc__ = """
RangeConfig is an abstract proto which can be used to describe ranges for different entities in TFX Pipeline.
"""

RollingRange.__doc__ = """
Describes a rolling range:
[most_recent_span - num_spans + 1,  most_recent_span].
For example, say you want the range to include only the latest span,
the appropriate RollingRange would simply be:
RollingRange <  num_spans = 1 >
The range is clipped based on available data.
ote that num_spans is required in RollingRange, while others are optional.
"""

StaticRange.__doc__ = """
Describes a static window within the specified span numbers [start_span_number, end_span_number].
Note that both numbers should be specified for StaticRange.
"""

TrainArgs = trainer_pb2.TrainArgs
EvalArgs = trainer_pb2.EvalArgs
del trainer_pb2

TrainArgs.__doc__ = """
Args specific to training in Traier.
"""

EvalArgs.__doc__ = """
Args specific to eval in Trainer.
"""

SplitsConfig = transform_pb2.SplitsConfig
del transform_pb2

SplitsConfig.__doc__ = """
Defines the splits config in Transform.
"""

TuneArgs = tuner_pb2.TuneArgs
del tuner_pb2

TuneArgs.__doc__ = """
Args specific to tuning in Tuner.
"""
