from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.params import Depends
from pydantic import BaseModel
import tritonclient.http as httpclient
import numpy as np
import uvicorn

app = FastAPI()

triton_client = httpclient.InferenceServerClient(url='localhost:8000', ssl=False)

@app.get("/infer")
def create_item():
    model_name = "simple"

    inputs = []
    outputs = []
    inputs.append(httpclient.InferInput("INPUT0", [1, 16], "INT32"))
    inputs.append(httpclient.InferInput("INPUT1", [1, 16], "INT32"))
    
    input0_data = np.arange(start=0, stop=16, dtype=np.int32)
    input0_data = np.expand_dims(input0_data, axis=0)
    input1_data = np.full(shape=(1, 16), fill_value=-1, dtype=np.int32)

    # Initialize the data
    inputs[0].set_data_from_numpy(input0_data, binary_data=False)
    inputs[1].set_data_from_numpy(input1_data, binary_data=True)

    outputs.append(httpclient.InferRequestedOutput("OUTPUT0", binary_data=True))
    outputs.append(httpclient.InferRequestedOutput("OUTPUT1", binary_data=False))
    query_params = {"test_1": 1, "test_2": 2}
    results = triton_client.infer(
        model_name,
        inputs,
        outputs=outputs,
        query_params=query_params,
        # headers=headers,
        # request_compression_algorithm=request_compression_algorithm,
        # response_compression_algorithm=response_compression_algorithm,
    )

    return results.get_response()
    


if __name__ == "__main__":
    uvicorn.run('main:app', host="127.0.0.1", port=8080, workers=8)