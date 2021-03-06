#!/usr/bin/env python3
from typing import Dict
from typing import List
from typing import Text

from custom_vision_client.models import Prediction
from custom_vision_client.prediction import PredictionClient
from custom_vision_client.prediction import PredictionConfig
from sanic import Sanic
from sanic.request import Request
from sanic.response import json
from sanic_cors import CORS

app = Sanic(__name__)
CORS(app)


def _classify(model_name: Text, image_url: Text) -> List[Prediction]:
    model = app.config.models.get(model_name)
    if not model:
        return []

    region = app.config.azure_region
    project_id = model['project_id']
    prediction_key = model['prediction_key']
    model_id = model.get('model_id')

    client = PredictionClient(PredictionConfig(
        region=region,
        project_id=project_id,
        prediction_key=prediction_key))

    predictions = client.classify_image(image_url, model_id)

    return predictions


def _to_dto(predictions: List[Prediction]) -> List[Dict]:
    return [{'score': _.Probability, 'name': _.Tag} for _ in predictions]


def _flatten(nested):
    return [item for sublist in nested for item in sublist]


@app.route('/v1/finish')
async def predict(request: Request):
    image_url = request.args.get('image_url')

    model_name = '__ROOT__'
    color_families = []
    finishes = []
    while True:
        predictions = _classify(model_name, image_url)
        if not predictions:
            break
        color_families.append(predictions)
        finishes = predictions
        model_name = max(predictions, key=lambda _: _.Probability).Tag

    return json({
        'color_families': _to_dto(_flatten(color_families[:-1])),
        'finishes': _to_dto(finishes),
    })


def _main():
    from argparse import ArgumentParser
    from json import loads
    from multiprocessing import cpu_count
    from os import getenv

    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--workers', type=int, default=cpu_count())
    args = parser.parse_args()

    app.config.azure_region = getenv('CVAI_REGION', 'southcentralus')
    app.config.models = loads(getenv('CVAI_MODELS', ''))
    app.run(host=args.host, port=args.port, workers=args.workers)


if __name__ == '__main__':
    _main()
