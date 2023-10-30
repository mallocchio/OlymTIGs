from tensorflow.keras.datasets import mnist
import numpy as np

def training(model):

    batch_size = 256
    nb_epoch = 10

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices    
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)


    # compiling
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    # trainig
    model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=nb_epoch, verbose=1)
    
    #evaluation
    score = model.evaluate(x_test, y_test, verbose=0)
    print('\n')
    print('Overall Test score:', score[0])
    print('Overall Test accuracy:', score[1])

    return model

def convert_tf_to_torch(tf_model, model_name):
    # input image dimensions
    img_rows, img_cols = 28, 28
    img_dim = img_rows * img_cols
    input_shape = (img_rows, img_cols, 1)

    tf_weights = tf_model.get_weights()

    # load pt state_dict
    net = Torch_LeNet1()
    sd = net.state_dict()


    # copy tf weights to pt
    def translate_convw(weights, index):
        convw = weights[index]
        convw = np.transpose(convw, (3, 2, 0, 1))
        convw = torch.from_numpy(convw)
        return convw


    def translate_bias(weights, index):
        convb = weights[index]
        convb = torch.from_numpy(convb)
        return convb


    sd['conv1.weight'] = translate_convw(tf_weights, 0)
    sd['conv1.bias'] = translate_bias(tf_weights, 1)
    sd['conv2.weight'] = translate_convw(tf_weights, 2)
    sd['conv2.bias'] = translate_bias(tf_weights, 3)

    out_w = tf_weights[4]
    out_w = np.transpose(out_w)
    out_w = torch.from_numpy(out_w)
    sd['out.weight'] = out_w

    sd['out.bias'] = translate_bias(tf_weights, 5)

    torch.save(sd, "./trained/" + f"{model_name}.pt")

def torch_load(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    return model
