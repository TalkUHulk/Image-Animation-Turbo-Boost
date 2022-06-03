# coding:utf-8
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import numpy as np


def do_inference(context, bindings, inputs, outputs, stream):
    """do_inference_v2 (for TensorRT 7.0+)

    This function is generalized for multiple inputs/outputs for full
    dimension networks.
    Inputs and outputs are expected to be lists of HostDeviceMem objects.
    """
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def get_input_shape(engine):
    binding = engine[0]
    assert engine.binding_is_input(binding)
    binding_dims = engine.get_binding_shape(binding)
    if len(binding_dims) == 4:
        return tuple(binding_dims[2:])
    elif len(binding_dims) == 3:
        return tuple(binding_dims[1:])
    else:
        raise ValueError('bad dims of binding %s: %s' % (binding, str(binding_dims)))


class HostDeviceMem(object):
    """Simple helper data class that's a little nicer to use than a 2-tuple."""

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

    def __del__(self):
        del self.device
        del self.host


def allocate_buffers(engine):
    """Allocates all host/device in/out buffers required for an engine."""
    inputs = []
    outputs = []
    bindings = []
    output_idx = 0
    stream = cuda.Stream()
    for binding in engine:
        binding_dims = engine.get_binding_shape(binding)
        if len(binding_dims) == 4:
            # explicit batch case (TensorRT 7+)
            size = trt.volume(binding_dims)
        elif len(binding_dims) == 3:
            # implicit batch case (TensorRT 6 or older)
            size = trt.volume(binding_dims) * engine.max_batch_size
        else:
            raise ValueError('bad dims of binding %s: %s' % (binding, str(binding_dims)))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            # each grid has 3 anchors, each anchor generates a detection
            # output of 7 float32 values
            outputs.append(HostDeviceMem(host_mem, device_mem))
            output_idx += 1
    return inputs, outputs, bindings, stream


class TrtModels(object):
    def _load_engine(self):
        with open(self.model, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def __call__(self, *args, **kwargs):
        return self.forward(*args)

    def __init__(self, model, cuda_ctx=None):
        """Initialize TensorRT plugins, engine and conetxt."""
        self.model = model
        self.cuda_ctx = cuda_ctx
        if self.cuda_ctx:
            self.cuda_ctx.push()

        self.inference_fn = do_inference
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.engine = self._load_engine()

        self.input_shape = get_input_shape(self.engine)

        try:
            self.context = self.engine.create_execution_context()
            self.inputs, self.outputs, self.bindings, self.stream = \
                allocate_buffers(self.engine)
        except Exception as e:
            raise RuntimeError('fail to allocate CUDA resources') from e
        finally:
            if self.cuda_ctx:
                self.cuda_ctx.pop()

    def __del__(self):
        """Free CUDA memories."""
        del self.outputs
        del self.inputs
        del self.stream

    def forward(self, *x):
        # Set host input to the image. The do_inference() function
        # will copy the input to the GPU before executing.
        for i in range(len(self.inputs)):
            self.inputs[i].host = np.ascontiguousarray(x[i])
        if self.cuda_ctx:
            self.cuda_ctx.push()
        trt_outputs = self.inference_fn(
            context=self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream)
        if self.cuda_ctx:
            self.cuda_ctx.pop()

        return trt_outputs


if __name__ == "__main__":
    # generator = TrtModels("../generator_sim.trt")
    #
    # source = np.random.random([1, 3, 256, 256]).astype(np.float32)
    #
    # driving_value = np.random.random([1, 10, 2]).astype(np.float32)
    # driving_jacobian = np.random.random([1, 10, 2, 2]).astype(np.float32)
    #
    # source_value = np.random.random([1, 10, 2]).astype(np.float32)
    # source_jacobian = np.random.random([1, 10, 2, 2]).astype(np.float32)
    #
    # out = generator.forward(source, driving_value, driving_jacobian, source_value, source_jacobian)
    # print(type(out), out[0].shape)

    import imageio
    from skimage.transform import resize

    kp_detector = TrtModels("../weights/kp_detector_sim.trt")
    source_image = imageio.imread("../Monalisa.png")
    source_image = resize(source_image, (256, 256))[..., :3]
    x_input = np.transpose(source_image[np.newaxis].astype(np.float32), (0, 3, 1, 2))
    kp_value, kp_jacobian = kp_detector(x_input.astype(np.float32))
