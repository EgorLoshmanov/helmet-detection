#!/usr/bin/env python3
"""Expose YOLOv8 boxes and class scores as separate ONNX outputs for INT8.

The original 1x6x8400 tensor combines box coordinates (~0..640) and class
probabilities (0..1). A single INT8 output scale rounds every probability to
zero, so the two existing inputs of the final Concat must be exported instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("models/helmet_detector.onnx"))
    parser.add_argument("--output", type=Path, default=Path("models/helmet_detector_split.onnx"))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Output exists: {args.output}")
    model = onnx.load(args.input)
    graph = model.graph
    final = graph.node[-1]
    if final.op_type != "Concat" or len(final.input) != 2 or list(final.output) != [graph.output[0].name]:
        raise SystemExit("Unexpected YOLO output graph; inspect before splitting")
    if len(final.attribute) != 1 or final.attribute[0].name != "axis" or final.attribute[0].i != 1:
        raise SystemExit("Expected a final channel-axis Concat")
    if len(graph.output) != 1 or [dim.dim_value for dim in graph.output[0].type.tensor_type.shape.dim] != [1, 6, 8400]:
        raise SystemExit("Expected original output shape 1x6x8400")
    boxes_name, scores_name = final.input
    graph.node.pop()
    graph.ClearField("output")
    graph.output.extend([
        helper.make_tensor_value_info(boxes_name, TensorProto.FLOAT, [1, 4, 8400]),
        helper.make_tensor_value_info(scores_name, TensorProto.FLOAT, [1, 2, 8400]),
    ])
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f"Saved {args.output}: boxes 1x4x8400, scores 1x2x8400")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
