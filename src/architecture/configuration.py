embedding = 768
dropout_rate = 0.1
number_of_heads = 12
number_of_groups = 4
context_length = 1024
bias = False
epsilon = 1e-6
vocabulary = 50257
layers = 12
embedding_expansion_rate = 4

gqa_configuration = {
    "embedding": embedding,
    "number_of_heads": number_of_heads,
    "number_of_groups": number_of_groups,
    "context_length": context_length,
    "dropout_rate": dropout_rate,
    "bias": bias,
}

ffn_configuration = {
    "embedding": embedding,
    "dropout_rate": dropout_rate,
    "bias": bias,
    "embedding_expansion_rate": embedding_expansion_rate
}

rmsn_configuration = {
    "embedding": embedding,
    "epsilon": epsilon
}

trf_configuration = {
    "gqa_configuration": gqa_configuration,
    "ffn_configuration": ffn_configuration,
    "rmsn_configuration": rmsn_configuration
}

model_configuration = {
    "embedding": embedding,
    "vocabulary": vocabulary,
    "dropout_rate": dropout_rate,
    "bias": bias,
    "layer": layers,
    "trf_configuration": trf_configuration,
    "rmsn_configuration": rmsn_configuration
}