import sys
import subprocess
import logging

from typing import Any, List, Text, Dict

import numpy as np
import tensorflow as tf
from openvino.inference_engine import IECore

from transformers.file_utils import cached_path

logger = logging.getLogger(__name__)


class OpenVINOModel:
    """A class that optimizes HuggingFace networks using Intel OpenVINO."""

    class _OutputWrapper:
        def __init__(self, value: np.ndarray) -> None:
            self.value = value

        def numpy(self) -> np.ndarray:
            return self.value

    def __init__(self, model_name: Text, config: Dict, model: Any = None) -> None:
        """Instantiates a new OpenVINO optimized model."""
        self.model = model
        self.model_name = model_name.replace("/", "_")
        self.ie = IECore()
        self.net = None
        self.exec_net = None
        self.out_name = ""
        # In case of dynamic input lenght, OpenVINO should reload network every time.
        # Due it is a time consuming procedure, there is an option to load a network
        # once for wider length and just pad inputs by zeros. Thanks to attention_mask,
        # output will match.
        self.max_length = config.get("openvino_max_length", 0)

        url = "https://huggingface.co/dkurt/openvino/resolve/main/" + self.model_name
        xml_path = cached_path(url + ".xml")
        bin_path = cached_path(url + ".bin")
        if xml_path is not None and bin_path is not None:
            import shutil

            shutil.copyfile(xml_path, xml_path + ".xml")
            self.net = self.ie.read_network(xml_path + ".xml", bin_path)

    def _load_model(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> None:
        # Serialize a Keras model
        @tf.function(
            input_signature=[
                {
                    "input_ids": tf.TensorSpec(
                        (None, None), tf.int32, name="input_ids"
                    ),
                    "attention_mask": tf.TensorSpec(
                        (None, None), tf.int32, name="attention_mask"
                    ),
                }
            ]
        )
        def serving(inputs: List[tf.TensorSpec]) -> tf.TensorSpec:
            output = self.model.call(inputs)
            return output[0]

        self.model.save("keras_model", signatures=serving)

        # Convert to OpenVINO IR
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mo",
                "--saved_model_dir=keras_model",
                "--model_name",
                self.model_name,
                "--input",
                "input_ids,attention_mask",
                "--input_shape",
                "{},{}".format(input_ids.shape, attention_mask.shape),
                "--disable_nhwc_to_nchw",
                "--data_type=FP16",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        proc.communicate()

        # Load model into memory
        self.net = self.ie.read_network(self.model_name + ".xml")

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
            self.exec_net = self.ie.load_network(self.net, "CPU")

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
