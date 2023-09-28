from pydantic import BaseModel


class EmbeddingRequest(BaseModel):

    """
    Base model class for incoming embedding request

    Attributes
    ----------
    texts : list[str]
        A list of texts to get embeddings for
    """

    texts: list[str]


class EmbeddingResponse(BaseModel):

    """
    Base model class for the response of the embedding endpoint

    Attributes
    ----------
    embeddings : list[list[float]]
    """
    embeddings: list[list[float]]
