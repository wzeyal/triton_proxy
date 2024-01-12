from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.params import Depends
from pydantic import BaseModel
import tritonclient.http as httpclient
import numpy as np
import uvicorn

app = FastAPI()

triton_client = httpclient.InferenceServerClient(url='172.17.0.1:8000', ssl=False, concurrency=3)

@app.get("/infer")
async def create_item():
    model_name = "sentiment_model"

    inputs = []
    outputs = []
    inputs.append(httpclient.InferInput("input_0", [1, 512], "INT64"))
    
    input0_data = np.arange(start=0, stop=512, dtype=np.int64)
    input0_data = np.expand_dims(input0_data, axis=0)

    # Initialize the data
    inputs[0].set_data_from_numpy(input0_data, binary_data=False)

    outputs.append(httpclient.InferRequestedOutput("output_0", binary_data=False))

    # query_params = {"test_1": 1, "test_2": 2}
    results = triton_client.infer(
        model_name,
        inputs,
        # outputs=outputs,
        # query_params=query_params,
        # headers=headers,
        # request_compression_algorithm=request_compression_algorithm,
        # response_compression_algorithm=response_compression_algorithm,
    )

    return results.get_response()
    


if __name__ == "__main__":
    uvicorn.run(
        'main:app',
        host="0.0.0.0",
        port=8080, workers=8
    )