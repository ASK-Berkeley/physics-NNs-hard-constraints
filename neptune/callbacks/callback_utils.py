import typing
from dataclasses import fields
import matplotlib.pyplot as plt
import jax

from ..types import CallbackPayload


def copy_payload(old_payload: CallbackPayload, **new_keys):
    replaced_keys = new_keys.keys()
    new_payload = {}
    for field in fields(old_payload):
        if field.name in replaced_keys:
            new_payload[field.name] = new_keys[field.name]
        else:
            new_payload[field.name] = old_payload.get_field(field.name)
    return CallbackPayload(**new_payload)


def split_payload_across_components(payload: CallbackPayload) -> typing.List[CallbackPayload]:
    # TODO Not actually sure if this still splits across components
    pde = payload.pde
    predicted = {'u': payload.predicted.grid()}
    target = payload.batch
    split_payloads = []
    for k in predicted.keys():
        split_payloads.append(copy_payload(
            payload, predicted=predicted[k], target=target[k], field_name=k))
    return split_payloads


def split_payload_across_batch(payload: CallbackPayload):
    predicted = payload.predicted
    batch_size = predicted.shape[0]
    split_payloads = []
    for i in range(batch_size):
        predicted = payload.predicted[i]
        batch = jax.tree_map(lambda x: x[i], payload.batch)
        target = payload.target
        if target is not None:
            target = target[i]
        split_payloads.append(copy_payload(
            payload, predicted=predicted, batch=batch, target=target, index=i))
    return split_payloads


def close_matplot_fig(fig):
    pass
