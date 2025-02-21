import tensorflow as tf

print("TensorFlow Version:", tf.__version__)

# List available physical GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU Available: {len(gpus)}")
    for gpu in gpus:
        print(f"  - {gpu}")
else:
    print("❌ No GPU detected.")

# Check if TensorFlow is using GPU
print("TensorFlow is using GPU:", tf.test.is_built_with_cuda())

# Run a test operation on GPU if available
if gpus:
    with tf.device("/GPU:0"):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[2.0, 0.0], [1.0, 2.0]])
        result = tf.matmul(a, b)
        print("Test TensorFlow GPU Computation Result:\n", result.numpy())

# Get detailed GPU information
from tensorflow.python.client import device_lib
print("\nAvailable Devices:")
print(device_lib.list_local_devices())

