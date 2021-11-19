import os
import sys
import shutil
import subprocess
import logging

from typing import Any, List, Text, Dict

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from openvino.inference_engine import IECore

from transformers.file_utils import hf_bucket_url, cached_path

logger = logging.getLogger(__name__)

ie = IECore()

# This is a global list of models with failed conversion.
# If model conversion failed before we won't call
# Model Optimizer once again because it might stuck CI for a long time.
failed_models = []


class OpenVINOModel:
    """A class that optimizes HuggingFace networks using Intel OpenVINO."""

    class _OutputWrapper:
        def __init__(self, value: np.ndarray) -> None:
            self.value = value

        def numpy(self) -> np.ndarray:
            return self.value

    def __init__(
        self, model: Any, config: Dict, model_name: Text, cache_dir: Text
    ) -> None:
        """Instantiates a new OpenVINO optimized model."""
        self.model = model
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.net = None
        self.exec_net = None
        self.out_name = ""
        # In case of dynamic input lenght, OpenVINO should reload network every time.
        # Due it is a time consuming procedure, there is an option to load a network
        # once for wider length and just pad inputs by zeros. Thanks to attention_mask,
        # output will match.
        self.max_length = config.get("openvino_max_length", 0)

    def _load_model(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> None:
        if self.model_name in failed_models:
            raise Exception("Model conversion failed before")

        # Check that model is in cache already
        url = hf_bucket_url(self.model_name, filename="tf_model.h5")
        path = cached_path(url, cache_dir=self.cache_dir)

        xml_path = path + ".xml"
        bin_path = path + ".bin"
        if not os.path.exists(xml_path) or not os.path.exists(bin_path):
            try:
                self._convert_model(path, input_ids, attention_mask)
            except Exception as e:
                logger.error(str(e))

            if not os.path.exists(xml_path):
                failed_models.append(self.model_name)
                raise Exception("Model conversion failed")

        # Load model into memory
        self.net = ie.read_network(xml_path)

    def _convert_model(
        self, tf_weights_path: Text, input_ids: np.ndarray, attention_mask: np.ndarray
    ) -> None:
        cache_dir = os.path.dirname(tf_weights_path)

        func = tf.function(lambda input_ids, attention_mask: self.model(input_ids, attention_mask=attention_mask))
        func = func.get_concrete_function(input_ids=tf.TensorSpec((None, None), tf.int32, name="input_ids"),
                                          attention_mask=tf.TensorSpec((None, None), tf.int32, name="attention_mask"))
        frozen_func = convert_variables_to_constants_v2(func)
        graph_def = frozen_func.graph.as_graph_def()

        pb_model_path = os.path.join(cache_dir, "frozen_graph.pb")
        with tf.io.gfile.GFile(pb_model_path, 'wb') as f:
            f.write(graph_def.SerializeToString())

        # Convert to OpenVINO IR
        subprocess.run(
            [
                sys.executable,
                "-m",
                "mo",
                "--output_dir",
                cache_dir,
                "--input_model",
                pb_model_path,
                "--model_name",
                os.path.basename(tf_weights_path),
                "--input",
                "input_ids,attention_mask",
                "--input_shape",
                "{},{}".format([1, input_ids.shape[1]], [1, attention_mask.shape[1]]),
                "--disable_nhwc_to_nchw",
                "--data_type",
                "FP32" if self.model_name == "distilbert-base-uncased" else "FP16",
            ],
            check=False,
        )

        os.remove(pb_model_path)

    def _init_model(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> None:
        # Reshape model in case of different input shape (batch is computed sequently)
        inputs_info = self.net.input_info
        if (
            inputs_info["input_ids"].input_data.shape[1] != input_ids.shape[1]
            or self.exec_net is None
        ):
            # Use batch size 1 because we process batch sequently.
            logger.info(f"Reshape model to 1x{input_ids.shape[1]}")
            self.net.reshape(
                {
                    "input_ids": [1, input_ids.shape[1]],
                    "attention_mask": [1, attention_mask.shape[1]],
                }
            )
            self.exec_net = None

        if self.exec_net is None:
            self.out_name = next(iter(self.net.outputs.keys()))
            self.exec_net = ie.load_network(self.net, "CPU")

    def _process_data(self, ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # In case of batching, we process samples one by one instead of
        # single forward pass. It is done because of heavy load_network step.
        batch_size = ids.shape[0]
        if batch_size > 1:
            out_shape = self.net.outputs[self.out_name].shape
            output = np.zeros([batch_size] + out_shape[1:], np.float32)
            for i in range(batch_size):
                out_i = self.exec_net.infer(
                    {"input_ids": ids[i : i + 1], "attention_mask": mask[i : i + 1],}
                )
                output[i] = out_i[self.out_name]
        else:
            output = self.exec_net.infer({"input_ids": ids, "attention_mask": mask})
            output = output[self.out_name]
        return output

    def __call__(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> List[Any]:
        """Does inference step in synchronous mode."""
        if self.net is None:
            self._load_model(input_ids, attention_mask)

        # If <max_length> specified, pad inputs by zeros
        batch_size, inp_length = input_ids.shape
        if inp_length < self.max_length:
            pad = ((0, 0), (0, self.max_length - inp_length))
            input_ids = np.pad(input_ids, pad)
            attention_mask = np.pad(attention_mask, pad)

        self._init_model(input_ids, attention_mask)

        output = self._process_data(input_ids, attention_mask)

        # Trunc padded values
        if inp_length != output.shape[1]:
            output = output[:, :inp_length]

        return [self._OutputWrapper(output)]
