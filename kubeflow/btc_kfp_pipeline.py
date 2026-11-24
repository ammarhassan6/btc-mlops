import os

from kfp import compiler, dsl

IMAGE = "ammarhassan1/btc-forecast-api:latest"
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "btc_kfp_pipeline.yaml")


@dsl.container_component
def fetch_data(raw_data: dsl.Output[dsl.Dataset]):
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["python", "-m", "src.etl_fetch"],
        args=["--out", raw_data.path],
    )


@dsl.container_component
def preprocess(raw_data: dsl.Input[dsl.Dataset], features: dsl.Output[dsl.Dataset]):
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["python", "-m", "src.preprocess"],
        args=["--input", raw_data.path, "--out", features.path],
    )


@dsl.container_component
def train(features: dsl.Input[dsl.Dataset], model: dsl.Output[dsl.Model]):
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["python", "-m", "src.train_forecast"],
        args=["--input", features.path, "--model-dir", model.path],
    )


@dsl.pipeline(
    name="btc-forecast-pipeline",
    description="Fetch BTC prices, build features and train the ARIMA model",
)
def btc_pipeline():
    fetch_task = fetch_data()
    # no inputs, so a cached result would reuse the first download forever
    fetch_task.set_caching_options(False)
    preprocess_task = preprocess(raw_data=fetch_task.outputs["raw_data"])
    train(features=preprocess_task.outputs["features"])


if __name__ == "__main__":
    compiler.Compiler().compile(btc_pipeline, OUTPUT_FILE)
    print(f"Pipeline compiled to {OUTPUT_FILE}")
