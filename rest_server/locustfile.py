from locust import HttpUser, task
import polars as pl


class QuickstartUser(HttpUser):

    dtypes = {
        "": pl.Int64,
        "Review_Date": pl.Utf8,
        "Author_Name": pl.Utf8,
        "Vehicle_Title": pl.Utf8,
        "Review_Title": pl.Utf8,
        "Review": pl.Utf8,
        "Rating": pl.Float64, }

    data = pl.read_csv("archive/*", dtypes=dtypes)

    @task
    def create_embeddings(self):

        texts = self.data.select("Review").sample(1).to_numpy().tolist()[0]

        self.client.post("text-embeddings", json={"texts": texts})
