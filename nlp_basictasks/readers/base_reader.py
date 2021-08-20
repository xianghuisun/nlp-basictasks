import torch

def convert_to_tensor(features):
    features_tensor={key:[] for key in features[0].__dict__.keys()}
    for feature in features:
        for key,value in feature.__dict__.items():
            features_tensor[key].append(value)
    features_tensor={key:torch.LongTensor(value) for key,value in features_tensor.items()}
    return features_tensor