python3 tensorflow-yolo-v3/convert_weights_pb.py --class_names signals.names --data_format NHWC --weights_file yolov3-tiny-signals.weights --tiny --data_type FP16
python3 /opt/intel/openvino_2019.1.133/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_darknet_yolov3_model.pb --tensorflow_use_custom_operations_config yolo_v3_tiny_custom.json --input_shape [1,416,416,3]

