from six.moves import cPickle

with open('pretrained_params.ckpt', 'rb') as f:
    conv_weights = cPickle.load(f)

print(type(conv_weights))
for key, value in conv_weights.items():
    print key, value.shape

print(type(conv_weights))
conv_weights = sorted(conv_weights.items(), key=lambda d:d[0])
for element in conv_weights:
    print element[0], element[1].shape