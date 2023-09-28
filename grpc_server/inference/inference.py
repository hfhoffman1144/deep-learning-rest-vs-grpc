from concurrent import futures
import grpc
from inference_pb2 import (
    InferenceResponse,
    Embedding
)
import inference_pb2_grpc
from sentence_transformers import SentenceTransformer

REQUESTED_DEIVCE = "mps"
MODEL = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2",
                            device=REQUESTED_DEIVCE)


class InferenceService(
    inference_pb2_grpc.InferenceServerServicer
):
    def inference(self, request, context):

        embeddings_list = MODEL.encode(request.texts).tolist()

        return InferenceResponse(embeddings=[Embedding(values=embedding) for embedding in embeddings_list])


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inference_pb2_grpc.add_InferenceServerServicer_to_server(
        InferenceService(), server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
