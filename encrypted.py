import tenseal as ts
import torch

def encrypt_chunk(args):
    ctx_serialized, x_chunk = args
    ctx = ts.context_from(ctx_serialized)
    return [ts.ckks_vector(ctx, x.tolist()).serialize() for x in x_chunk]

def evaluate_chunk(args):
    ctx_serialized, weight_serialized, bias_serialized, enc_x_chunk, y_chunk = args

    ctx = ts.context_from(ctx_serialized)
    weight = ts.ckks_vector_from(ctx, weight_serialized)
    bias = ts.ckks_vector_from(ctx, bias_serialized)

    outputs = []
    for enc_x_ser in enc_x_chunk:
        enc_x = ts.ckks_vector_from(ctx, enc_x_ser)
        out = enc_x.dot(weight) + bias
        outputs.append(out.serialize())
    return outputs, y_chunk