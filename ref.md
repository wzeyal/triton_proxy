"Hyper-arguments" :
- sync/async in fast api
- sync/async in tritonclient infer
- unicorn/gunicorn
- instance_group [  {    count: 8  } ]
- max_batch_size
- concurrency
- dynamic_batching {  max_queue_delay_microseconds: 100  }


simple, batch 8, 

max_batch_size: 128

instance_group [
  {
    count: 8
  }
]

 async_simple, users=4000, 218/4000
RPS = 300~400
latency = 10000


# https://github.com/triton-inference-server/server/issues/5205

# https://developer.nvidia.com/blog/fast-and-scalable-ai-model-deployment-with-nvidia-triton-inference-server/

# https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#delayed-batching

# https://towardsdatascience.com/hugging-face-transformer-inference-under-1-millisecond-latency-e1be0057a51c

# https://blog.ml6.eu/triton-ensemble-model-for-deploying-transformers-into-production-c0f727c012e3

# https://www.youtube.com/watch?v=P7dvC31Ggxk&t=2262s

https://developer.nvidia.com/blog/identifying-the-best-ai-model-serving-configurations-at-scale-with-triton-model-analyzer/


with tritonhttpclient.InferenceServerClient(url=url, verbose=False, concurrency=32) as client:
    ...
    
    # Hit triton server
    n_requests = 4
    responses = []
    for i in range(n_requests):
        responses.append(client.async_infer(model_name, model_version=model_version, inputs=inputs, outputs=outputs))