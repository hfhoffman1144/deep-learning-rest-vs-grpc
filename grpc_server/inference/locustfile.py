import polars as pl
import gevent
import grpc_user
import inference_pb2
import inference_pb2_grpc
from inference import serve
from locust import events, task

dtypes = {
    "": pl.Int64,
    "Review_Date": pl.Utf8,
    "Author_Name": pl.Utf8,
    "Vehicle_Title": pl.Utf8,
    "Review_Title": pl.Utf8,
    "Review": pl.Utf8,
    "Rating": pl.Float64,
}

data = pl.read_csv("archive/*", dtypes=dtypes)

# Start the dummy server. This is not something you would do in a real test.


@events.init.add_listener
def run_grpc_server(environment, **_kwargs):
    gevent.spawn(serve)


class InferenceGrpcUser(grpc_user.GrpcUser):
    host = "localhost:50051"
    stub_class = inference_pb2_grpc.InferenceServerStub

    @task
    def sayHello(self):
        text = data.select("Review").sample(1).to_numpy()[0]
        self.stub.inference(inference_pb2.InferenceRequest(texts=text))
